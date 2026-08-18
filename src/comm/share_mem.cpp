#include "comm.h"
#include "../ctl_path.h"

Comm::Comm(pid_t pid) {
}

Comm::~Comm() {
}

void Comm::setup() {
}

void Comm::send_msg(uint32_t msg) {
}

uint32_t Comm::recv_msg() {
    return 0;
}

bool Comm::is_finished() {
    return false;
}

ShareMemComm::ShareMemComm(pid_t pid) : Comm(pid), pid(pid) {
}

ShareMemComm::~ShareMemComm() {
}

void ShareMemComm::setup() {
    char control_name[512];
    bool ctl_mode = false;
    const char* ctl_dir = gpu_cr::CtlDir(&ctl_mode);
    snprintf(control_name, sizeof(control_name), "%s/control-%d", ctl_dir, pid);
    // 0777: the coordinator (agent container) and the workload run as
    // different users in different containers but share this file.
    // Previously applied as a Dockerfile.build sed patch; now source truth.
    fd_control = open(control_name, O_CREAT | O_RDWR, 0777);
    if (fd_control < 0) {
        perror("open()");
        exit(EXIT_FAILURE);
    }
    fchmod(fd_control, 0777);
    // Set file size before mmap to avoid Bus error
    if (ftruncate(fd_control, HUGE_PAGE_SIZE) < 0) {
        perror("ftruncate()");
        exit(EXIT_FAILURE);
    }
    if (ctl_mode) {
        // tmpfs reserves nothing at ftruncate or mmap: without this, a full
        // ctl tmpfs surfaces as SIGBUS at the first store — worst case
        // inside the .so's signal handler. Allocate only the stored extent;
        // the sparse 2MiB tail keeps the legacy file layout for free.
        // cr_client runs setup() first, so ENOSPC lands here, in the
        // coordinator, as a clean exit.
        int rc = posix_fallocate(
            fd_control, 0,
            static_cast<off_t>(gpu_cr::RoundUp4K(sizeof(signal_controls))));
        if (rc != 0) {
            fprintf(stderr, "posix_fallocate(%s): %s (ctl tmpfs full?)\n", control_name, strerror(rc));
            exit(EXIT_FAILURE);
        }
    }
    control = (signal_controls*)mmap(NULL, HUGE_PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_control, 0);
    if (control == MAP_FAILED) {
        perror("mmap()");
        exit(EXIT_FAILURE);
    }
}

bool ShareMemComm::is_finished() {
    return recv_msg() == FINISH_MSG;
}

void ShareMemComm::send_msg(uint32_t msg) {
    control->signal = msg;
}

uint32_t ShareMemComm::recv_msg() {
    return control->signal;
}