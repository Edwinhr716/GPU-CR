// Upstream-baseline UDS fd exchange (ipc_fd_exchange.cpp): server
// lifecycle, SCM_RIGHTS round trip of real fds, and the documented -1
// error returns. Runs against this process's own server. Linux-only.

#include "ipc_fd_exchange.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>

#include "gtest/gtest.h"

namespace gpu_cr {
namespace {

std::string SockPath(pid_t pid) {
  return "/tmp/cr_ipc_fd_" + std::to_string(pid) + ".sock";
}

// RAII so a failing assertion cannot leak the server thread into later
// tests (or into a death-test fork).
struct ServerGuard {
  ServerGuard() { rc = uds_fd_server_start(); }
  ~ServerGuard() { uds_fd_server_stop(); }
  int rc;
};

ino_t InodeOf(int fd) {
  struct stat st;
  if (fstat(fd, &st) != 0) return 0;
  return st.st_ino;
}

TEST(IpcFdExchangeTest, RejectsInvalidCounts) {
  int out[1];
  int fds[1] = {0};
  EXPECT_EQ(uds_receive_fds(getpid(), fds, 0, out), -1);
  EXPECT_EQ(uds_receive_fds(getpid(), fds, 65, out), -1);  // > UDS_MAX_FDS
}

TEST(IpcFdExchangeTest, FailsWhenPeerHasNoServer) {
  const pid_t bogus_pid = 999999999;
  unlink(SockPath(bogus_pid).c_str());
  int fds[1] = {0};
  int out[1];
  EXPECT_EQ(uds_receive_fds(bogus_pid, fds, 1, out), -1);
}

TEST(IpcFdExchangeTest, ServerLifecycle) {
  {
    ServerGuard server;
    ASSERT_EQ(server.rc, 0);

    struct stat st;
    ASSERT_EQ(stat(SockPath(getpid()).c_str(), &st), 0);
    EXPECT_TRUE(S_ISSOCK(st.st_mode));

    // Second start while running is a no-op success.
    EXPECT_EQ(uds_fd_server_start(), 0);
  }
  // Stop removed the socket file; a second stop is a no-op.
  struct stat st;
  EXPECT_NE(stat(SockPath(getpid()).c_str(), &st), 0);
  uds_fd_server_stop();
}

TEST(IpcFdExchangeTest, RoundTripDuplicatesFd) {
  char path[] = "/tmp/gpu-cr-fdx-XXXXXX";
  int fd = mkstemp(path);
  ASSERT_GE(fd, 0);
  const char payload[] = "gpu-cr fd exchange";
  ASSERT_EQ(write(fd, payload, sizeof(payload)),
            static_cast<ssize_t>(sizeof(payload)));

  ServerGuard server;
  ASSERT_EQ(server.rc, 0);

  int out = -1;
  ASSERT_EQ(uds_receive_fds(getpid(), &fd, 1, &out), 1);
  ASSERT_GE(out, 0);
  EXPECT_NE(out, fd);  // a new descriptor, not the requested number

  // SCM_RIGHTS installs a descriptor for the same file.
  EXPECT_EQ(InodeOf(out), InodeOf(fd));
  char got[sizeof(payload)] = "";
  ASSERT_EQ(pread(out, got, sizeof(payload), 0),
            static_cast<ssize_t>(sizeof(payload)));
  EXPECT_STREQ(got, payload);

  close(out);
  close(fd);
  unlink(path);
}

TEST(IpcFdExchangeTest, TransfersMultipleFdsInOneRequest) {
  char path_a[] = "/tmp/gpu-cr-fdx-a-XXXXXX";
  char path_b[] = "/tmp/gpu-cr-fdx-b-XXXXXX";
  int fds[2] = {mkstemp(path_a), mkstemp(path_b)};
  ASSERT_GE(fds[0], 0);
  ASSERT_GE(fds[1], 0);

  ServerGuard server;
  ASSERT_EQ(server.rc, 0);

  int out[2] = {-1, -1};
  ASSERT_EQ(uds_receive_fds(getpid(), fds, 2, out), 2);
  EXPECT_EQ(InodeOf(out[0]), InodeOf(fds[0]));
  EXPECT_EQ(InodeOf(out[1]), InodeOf(fds[1]));

  for (int i = 0; i < 2; i++) {
    close(out[i]);
    close(fds[i]);
  }
  unlink(path_a);
  unlink(path_b);
}

// The server keeps static state (socket fd, path, thread); a
// stop-then-start cycle on the same PID must serve a fresh round trip,
// pinning the reset a later lifecycle refactor could break.
TEST(IpcFdExchangeTest, RestartAfterStopServesAgain) {
  { ServerGuard first; ASSERT_EQ(first.rc, 0); }

  char path[] = "/tmp/gpu-cr-fdx-restart-XXXXXX";
  int fd = mkstemp(path);
  ASSERT_GE(fd, 0);

  ServerGuard second;
  ASSERT_EQ(second.rc, 0);
  int out = -1;
  ASSERT_EQ(uds_receive_fds(getpid(), &fd, 1, &out), 1);
  EXPECT_EQ(InodeOf(out), InodeOf(fd));

  close(out);
  close(fd);
  unlink(path);
}

// Requesting a descriptor the peer does not hold must fail the call, not
// hang or hand back garbage (the kernel refuses to SCM_RIGHTS a bad fd).
TEST(IpcFdExchangeTest, InvalidPeerFdFailsCleanly) {
  // A high fd number that is free in this process; the client's own
  // transient socket takes a low number, so this stays free.
  int bogus = 973;
  while (bogus < 1000 && fcntl(bogus, F_GETFD) >= 0) bogus++;
  ASSERT_LT(bogus, 1000) << "no free high fd found";

  ServerGuard server;
  ASSERT_EQ(server.rc, 0);

  int out = -1;
  EXPECT_EQ(uds_receive_fds(getpid(), &bogus, 1, &out), -1);
}

}  // namespace
}  // namespace gpu_cr
