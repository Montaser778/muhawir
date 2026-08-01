
---

## What actually happened when deploying

Fly.io was tried first and failed, and the failure is worth recording
because the symptom was misleading.

Signalling succeeded every time: the SDP offer was answered, the peer
connection initialised, and the browser reported `CONNECTING`. Then the
call ended with no audio in either direction. Adding a UDP service to
`fly.toml` did not help, and neither did routing the browser through a
TURN relay.

The cause is that aiortc gathers ICE candidates from the interfaces it
can see, which inside a Fly machine are container-internal addresses no
external browser can reach. TURN relayed the browser side only; the
server still advertised addresses that went nowhere.

A Cloudflare quick tunnel to a locally running server worked on the first
attempt, with a median response latency of 623 ms — better than the
800 ms target. The server was on a real host with a real address, which
is the whole difference.

Conclusion: this application needs a host with a directly reachable IP.
A managed platform that sits a network layer between the container and
the internet will break WebRTC media, whatever the port configuration says.
