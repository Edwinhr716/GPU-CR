#ifndef COMM_H
#define COMM_H
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "../common.h"

class Comm {
public:
    Comm(pid_t pid);
    virtual ~Comm();
    virtual void setup();

    virtual void send_msg(uint32_t msg);
    virtual uint32_t recv_msg();

    virtual bool is_finished();
};

#define INIT_MSG 10
#define CKPT_MSG 11
#define RESTORE_MSG 12
#define FINISH_MSG 0

class ShareMemComm : public Comm {
public:
    signal_controls* control;
    pid_t pid;

    ShareMemComm(pid_t pid);
    ~ShareMemComm();
    void setup() override;

    void send_msg(uint32_t msg) override;
    uint32_t recv_msg() override;

    bool is_finished() override;
};

#endif