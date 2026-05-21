#include "comm.h"

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
    const char* ctl_dir = std::getenv("EXPORT_FILE_PATH");
    if (!ctl_dir) ctl_dir = "/mnt/huge-ckpt";
    snprintf(control_name, sizeof(control_name), "%s/control-%d", ctl_dir, pid);
    int fd_control = open(control_name, O_CREAT | O_RDWR, 0755);
    if (fd_control < 0) {
        perror("open()");
        exit(EXIT_FAILURE);
    }
    // Set file size before mmap to avoid Bus error
    if (ftruncate(fd_control, HUGE_PAGE_SIZE) < 0) {
        perror("ftruncate()");
        exit(EXIT_FAILURE);
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