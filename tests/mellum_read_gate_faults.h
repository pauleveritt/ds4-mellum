#include <poll.h>
int mellum_gate_test_poll(struct pollfd *, nfds_t, int);
#define MELLUM_GATE_POLL mellum_gate_test_poll
#define MELLUM_GATE_TIMEOUT_MS 40
