/**
 * ipc_fd_exchange.cpp - Cross-process fd exchange via Unix Domain Sockets (SCM_RIGHTS)
 *
 * Replaces the broken pidfd_getfd() + /proc/pid/fd/ approach.
 * CUDA shareable handles (cuMem VMM IPC fds) can ONLY be transferred
 * between processes using SCM_RIGHTS — open() on /proc/pid/fd/N creates
 * a new fd to the device node but loses the CUDA memory handle metadata.
 *
 * Protocol:
 *   Server side (exporting process):
 *     1. uds_fd_server_start() creates /tmp/cr_ipc_fd_<pid>.sock
 *     2. Background thread accepts one connection
 *     3. Reads a request: {num_fds, fd_list[]}
 *     4. Sends the requested fds via SCM_RIGHTS (sendmsg with ancillary data)
 *     5. Closes connection, loops for next (supports multiple importers)
 *
 *   Client side (importing process):
 *     1. uds_receive_fds() connects to /tmp/cr_ipc_fd_<peer_pid>.sock
 *     2. Sends request: {num_fds, fd_list[]}
 *     3. Receives fds via SCM_RIGHTS (recvmsg)
 *     4. Returns local fd numbers
 */

#include "ipc_fd_exchange.h"

#include <cstdio>
#include <cerrno>
#include <cstring>
#include <alloca.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/syscall.h>

// Max fds per single SCM_RIGHTS transfer
#define UDS_MAX_FDS 64

// ---------------------------------------------------------------------------
// Socket path helper
// ---------------------------------------------------------------------------
static void make_sock_path(char* buf, size_t buflen, pid_t pid) {
    snprintf(buf, buflen, "/tmp/cr_ipc_fd_%d.sock", pid);
}

// ---------------------------------------------------------------------------
// Server side: serve fds via SCM_RIGHTS
// ---------------------------------------------------------------------------

struct UdsServerCtx {
    int listen_fd;
    volatile int running;
    pthread_t thread;
    char sock_path[128];
};

static UdsServerCtx g_server = { -1, 0, 0, {0} };

/**
 * Send an array of file descriptors over a UDS connection using SCM_RIGHTS.
 */
static int send_fds(int conn_fd, const int* fds, int num_fds) {
    // We send a single byte as the message payload (required by sendmsg)
    char dummy = 'F';
    struct iovec iov;
    iov.iov_base = &dummy;
    iov.iov_len = 1;

    // Ancillary data buffer for SCM_RIGHTS
    size_t cmsg_space = CMSG_SPACE(sizeof(int) * num_fds);
    char* cmsg_buf = (char*)alloca(cmsg_space);
    memset(cmsg_buf, 0, cmsg_space);

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf;
    msg.msg_controllen = cmsg_space;

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int) * num_fds);
    memcpy(CMSG_DATA(cmsg), fds, sizeof(int) * num_fds);

    ssize_t sent = sendmsg(conn_fd, &msg, 0);
    if (sent < 0) {
        fprintf(stderr, "[FD-SERVER] sendmsg failed: %s (errno=%d)\n",
                strerror(errno), errno);
        return -1;
    }
    return 0;
}

/**
 * Server thread: accept connections, read fd requests, send fds via SCM_RIGHTS.
 */
static void* uds_server_thread(void* arg) {
    (void)arg;
    fprintf(stderr, "[FD-SERVER] Server thread started (pid=%d, sock=%s)\n",
            getpid(), g_server.sock_path);

    while (g_server.running) {
        // Use poll-like approach: accept with a timeout via non-blocking + sleep
        struct sockaddr_un peer_addr;
        socklen_t peer_len = sizeof(peer_addr);

        // Set a short timeout using SO_RCVTIMEO on the listen socket
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 50000; // 50ms (reduced from 200ms for faster shutdown)
        setsockopt(g_server.listen_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        int conn_fd = accept(g_server.listen_fd, (struct sockaddr*)&peer_addr, &peer_len);
        if (conn_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue; // timeout, check running flag
            }
            if (!g_server.running) break;
            fprintf(stderr, "[FD-SERVER] accept failed: %s\n", strerror(errno));
            continue;
        }

        fprintf(stderr, "[FD-SERVER] Accepted connection (conn_fd=%d)\n", conn_fd);

        // Read request: first 4 bytes = num_fds, then num_fds * 4 bytes = fd numbers
        int num_fds = 0;
        ssize_t n = recv(conn_fd, &num_fds, sizeof(num_fds), MSG_WAITALL);
        if (n != sizeof(num_fds) || num_fds <= 0 || num_fds > UDS_MAX_FDS) {
            fprintf(stderr, "[FD-SERVER] Invalid request: n=%zd num_fds=%d\n", n, num_fds);
            close(conn_fd);
            continue;
        }

        int requested_fds[UDS_MAX_FDS];
        n = recv(conn_fd, requested_fds, sizeof(int) * num_fds, MSG_WAITALL);
        if (n != (ssize_t)(sizeof(int) * num_fds)) {
            fprintf(stderr, "[FD-SERVER] Short read on fd list: got %zd, expected %zu\n",
                    n, sizeof(int) * num_fds);
            close(conn_fd);
            continue;
        }

        fprintf(stderr, "[FD-SERVER] Request: %d fds [", num_fds);
        for (int i = 0; i < num_fds; i++)
            fprintf(stderr, "%s%d", i ? "," : "", requested_fds[i]);
        fprintf(stderr, "]\n");

        // Validate that all requested fds are valid in this process
        int valid_fds[UDS_MAX_FDS];
        int valid_count = 0;
        for (int i = 0; i < num_fds; i++) {
            if (fcntl(requested_fds[i], F_GETFD) >= 0) {
                valid_fds[valid_count++] = requested_fds[i];
            } else {
                fprintf(stderr, "[FD-SERVER] WARNING: requested fd %d is invalid in this process\n",
                        requested_fds[i]);
                // Send -1 as placeholder — client will detect this
                valid_fds[valid_count++] = requested_fds[i]; // still send it, client handles error
            }
        }

        // Send the count of fds we're about to send
        int send_count = num_fds;
        send(conn_fd, &send_count, sizeof(send_count), 0);

        // Send the fds via SCM_RIGHTS
        if (send_fds(conn_fd, requested_fds, num_fds) < 0) {
            fprintf(stderr, "[FD-SERVER] Failed to send fds\n");
        } else {
            fprintf(stderr, "[FD-SERVER] Sent %d fds successfully\n", num_fds);
        }

        close(conn_fd);
    }

    fprintf(stderr, "[FD-SERVER] Server thread exiting\n");
    return nullptr;
}

int uds_fd_server_start() {
    if (g_server.running) {
        fprintf(stderr, "[FD-SERVER] Server already running\n");
        return 0;
    }

    pid_t my_pid = getpid();
    make_sock_path(g_server.sock_path, sizeof(g_server.sock_path), my_pid);

    // Remove stale socket file
    unlink(g_server.sock_path);

    // Create UDS listener
    g_server.listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (g_server.listen_fd < 0) {
        fprintf(stderr, "[FD-SERVER] socket() failed: %s\n", strerror(errno));
        return -1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, g_server.sock_path, sizeof(addr.sun_path) - 1);

    if (bind(g_server.listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[FD-SERVER] bind(%s) failed: %s\n",
                g_server.sock_path, strerror(errno));
        close(g_server.listen_fd);
        g_server.listen_fd = -1;
        return -1;
    }

    if (listen(g_server.listen_fd, 8) < 0) {
        fprintf(stderr, "[FD-SERVER] listen() failed: %s\n", strerror(errno));
        close(g_server.listen_fd);
        unlink(g_server.sock_path);
        g_server.listen_fd = -1;
        return -1;
    }

    // Make socket world-accessible (both processes run as same user, but just in case)
    chmod(g_server.sock_path, 0777);

    g_server.running = 1;

    if (pthread_create(&g_server.thread, nullptr, uds_server_thread, nullptr) != 0) {
        fprintf(stderr, "[FD-SERVER] pthread_create failed: %s\n", strerror(errno));
        g_server.running = 0;
        close(g_server.listen_fd);
        unlink(g_server.sock_path);
        g_server.listen_fd = -1;
        return -1;
    }

    fprintf(stderr, "[FD-SERVER] Started UDS server at %s (pid=%d)\n",
            g_server.sock_path, my_pid);
    return 0;
}

void uds_fd_server_stop() {
    if (!g_server.running) return;

    fprintf(stderr, "[FD-SERVER] Stopping server...\n");
    g_server.running = 0;

    // Close listen socket to unblock accept()
    if (g_server.listen_fd >= 0) {
        close(g_server.listen_fd);
        g_server.listen_fd = -1;
    }

    pthread_join(g_server.thread, nullptr);

    // Clean up socket file
    if (g_server.sock_path[0]) {
        unlink(g_server.sock_path);
        g_server.sock_path[0] = '\0';
    }

    fprintf(stderr, "[FD-SERVER] Server stopped\n");
}

// ---------------------------------------------------------------------------
// Client side: request fds from peer via SCM_RIGHTS
// ---------------------------------------------------------------------------

/**
 * Receive file descriptors from a UDS connection using SCM_RIGHTS.
 */
static int recv_fds(int conn_fd, int* fds, int num_fds) {
    char dummy;
    struct iovec iov;
    iov.iov_base = &dummy;
    iov.iov_len = 1;

    size_t cmsg_space = CMSG_SPACE(sizeof(int) * num_fds);
    char* cmsg_buf = (char*)alloca(cmsg_space);
    memset(cmsg_buf, 0, cmsg_space);

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf;
    msg.msg_controllen = cmsg_space;

    ssize_t received = recvmsg(conn_fd, &msg, 0);
    if (received < 0) {
        fprintf(stderr, "[FD-CLIENT] recvmsg failed: %s (errno=%d)\n",
                strerror(errno), errno);
        return -1;
    }

    // Extract fds from ancillary data
    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) {
        fprintf(stderr, "[FD-CLIENT] No SCM_RIGHTS in response\n");
        return -1;
    }

    int received_count = (cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int);
    if (received_count != num_fds) {
        fprintf(stderr, "[FD-CLIENT] Expected %d fds, got %d\n", num_fds, received_count);
    }

    int copy_count = (received_count < num_fds) ? received_count : num_fds;
    memcpy(fds, CMSG_DATA(cmsg), sizeof(int) * copy_count);

    return copy_count;
}

int uds_receive_fds(pid_t peer_pid, const int* peer_fds, int num_fds, int* local_fds) {
    if (num_fds <= 0 || num_fds > UDS_MAX_FDS) {
        fprintf(stderr, "[FD-CLIENT] Invalid num_fds=%d\n", num_fds);
        return -1;
    }

    char sock_path[128];
    make_sock_path(sock_path, sizeof(sock_path), peer_pid);

    fprintf(stderr, "[FD-CLIENT] Connecting to peer %d at %s for %d fds\n",
            peer_pid, sock_path, num_fds);

    // Connect to peer's UDS server with retries (server may not be ready yet)
    int sock_fd = -1;
    for (int attempt = 0; attempt < 20; attempt++) {
        sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (sock_fd < 0) {
            fprintf(stderr, "[FD-CLIENT] socket() failed: %s\n", strerror(errno));
            return -1;
        }

        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);

        if (connect(sock_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            break; // connected
        }

        close(sock_fd);
        sock_fd = -1;

        if (attempt < 19) {
            usleep(10000); // 10ms between retries (reduced from 100ms)
        }
    }

    if (sock_fd < 0) {
        fprintf(stderr, "[FD-CLIENT] Failed to connect to %s after retries\n", sock_path);
        return -1;
    }

    fprintf(stderr, "[FD-CLIENT] Connected to peer %d\n", peer_pid);

    // Send request: num_fds + fd list
    if (send(sock_fd, &num_fds, sizeof(num_fds), 0) != sizeof(num_fds)) {
        fprintf(stderr, "[FD-CLIENT] Failed to send num_fds\n");
        close(sock_fd);
        return -1;
    }
    if (send(sock_fd, peer_fds, sizeof(int) * num_fds, 0) != (ssize_t)(sizeof(int) * num_fds)) {
        fprintf(stderr, "[FD-CLIENT] Failed to send fd list\n");
        close(sock_fd);
        return -1;
    }

    // Receive confirmation: how many fds the server will send
    int send_count = 0;
    ssize_t n = recv(sock_fd, &send_count, sizeof(send_count), MSG_WAITALL);
    if (n != sizeof(send_count) || send_count <= 0) {
        fprintf(stderr, "[FD-CLIENT] Invalid send_count response: n=%zd count=%d\n",
                n, send_count);
        close(sock_fd);
        return -1;
    }

    // Receive fds via SCM_RIGHTS
    int received = recv_fds(sock_fd, local_fds, send_count);
    close(sock_fd);

    if (received > 0) {
        fprintf(stderr, "[FD-CLIENT] Received %d fds from peer %d: [", received, peer_pid);
        for (int i = 0; i < received; i++)
            fprintf(stderr, "%s%d", i ? "," : "", local_fds[i]);
        fprintf(stderr, "]\n");
    }

    return received;
}

// ---------------------------------------------------------------------------
// Legacy pidfd_copy_fd (kept for reference, NOT used for CUDA handles)
// ---------------------------------------------------------------------------

#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif

#ifndef SYS_pidfd_getfd
#define SYS_pidfd_getfd 438
#endif

int pidfd_copy_fd(pid_t target_pid, int target_fd) {
    fprintf(stderr, "[FD-EXCHANGE] WARNING: pidfd_copy_fd called — this does NOT work "
            "for CUDA shareable handles! Use uds_receive_fds() instead.\n");

    int pidfd = (int)syscall(SYS_pidfd_open, target_pid, 0);
    if (pidfd < 0) {
        fprintf(stderr, "[FD-EXCHANGE] pidfd_open(pid=%d) failed: %s (errno=%d)\n",
                target_pid, strerror(errno), errno);
        return -1;
    }

    int local_fd = (int)syscall(SYS_pidfd_getfd, pidfd, target_fd, 0);
    close(pidfd);

    if (local_fd < 0) {
        fprintf(stderr, "[FD-EXCHANGE] pidfd_getfd(pid=%d, fd=%d) failed: %s (errno=%d)\n",
                target_pid, target_fd, strerror(errno), errno);
        return -1;
    }

    fprintf(stderr, "[FD-EXCHANGE] Copied fd %d from pid %d → local fd %d\n",
            target_fd, target_pid, local_fd);
    return local_fd;
}
