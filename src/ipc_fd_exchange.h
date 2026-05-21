/**
 * ipc_fd_exchange.h - Cross-process fd exchange via Unix Domain Sockets (SCM_RIGHTS)
 *
 * Transfers CUDA shareable file descriptors between processes using
 * sendmsg/recvmsg with SCM_RIGHTS ancillary data. This is the only
 * reliable way to transfer CUDA cuMem IPC handles between processes —
 * pidfd_getfd() requires CAP_SYS_PTRACE and /proc/pid/fd/ open() does
 * NOT preserve CUDA handle metadata.
 *
 * Protocol:
 *   1. Exporting process calls uds_fd_server_start() after re-export
 *      → creates /tmp/cr_ipc_fd_<pid>.sock, spawns listener thread
 *   2. Importing process calls uds_receive_fds() for each peer
 *      → connects to peer's UDS, sends requested fd list, receives
 *        real fds via SCM_RIGHTS
 *   3. After import phase, each process calls uds_fd_server_stop()
 */

#ifndef IPC_FD_EXCHANGE_H
#define IPC_FD_EXCHANGE_H

#include <sys/types.h>

/**
 * Start a UDS server that serves this process's file descriptors.
 * Creates socket at /tmp/cr_ipc_fd_<pid>.sock and spawns a background
 * thread to accept connections and send fds via SCM_RIGHTS.
 *
 * @return 0 on success, -1 on error
 */
int uds_fd_server_start();

/**
 * Stop the UDS server and clean up the socket file.
 * Blocks until the server thread exits.
 */
void uds_fd_server_stop();

/**
 * Request file descriptors from a peer process via its UDS server.
 *
 * Connects to /tmp/cr_ipc_fd_<peer_pid>.sock, sends the list of
 * fd numbers to request, and receives the actual fds via SCM_RIGHTS.
 *
 * @param peer_pid    PID of the peer process running a UDS server
 * @param peer_fds    Array of fd numbers in the peer process to request
 * @param num_fds     Number of fds to request
 * @param local_fds   Output array to receive the local fd numbers
 * @return number of fds successfully received, or -1 on error
 */
int uds_receive_fds(pid_t peer_pid, const int* peer_fds, int num_fds, int* local_fds);

/**
 * [DEPRECATED] Copy a file descriptor from target_pid's fd table.
 * This function is kept for reference but should NOT be used for CUDA
 * shareable handles — it falls back to /proc/pid/fd/ which does not
 * preserve CUDA handle metadata.
 */
int pidfd_copy_fd(pid_t target_pid, int target_fd);

#endif // IPC_FD_EXCHANGE_H
