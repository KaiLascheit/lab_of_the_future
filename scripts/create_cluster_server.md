# Aria Gen 2 → Imperial CX3: live frame forwarding setup

This README documents the **working network path** from the Ubuntu laptop (**Megatron**) to a Python HTTP receiver on an **Imperial College London CX3 GPU compute node**. We tested the path successfully by uploading a local `hello.txt` file through an SSH tunnel.

**Current status:** HPC receiver and laptop-to-cluster file transfer work. Forwarding *live Aria frames* using the laptop's existing Python RGB callback is the next integration step; the test here does not yet demonstrate continuous frame streaming or VLM inference.

## Architecture

```text
Aria Gen 2 ──Wi-Fi──> Megatron (Aria Python SDK, RGB frames)
                         │
                         │ HTTP POST to 127.0.0.1:8001/frame
                         ▼
                    Local SSH tunnel
                         │ encrypted SSH
                         ▼
            Imperial CX3 login node (login-ai)
                         │ cluster network / IPv6
                         ▼
             Allocated CX3 GPU compute node
                         │ Uvicorn/FastAPI, [::]:8000
                         ▼
                ~/aria_received_frames/

Future: replace/store-and-forward with VLM inference on the allocated GPU.
```

## Prerequisites

- Laptop can SSH to Imperial CX3 (Imperial network or approved VPN, as applicable).
- Imperial HPC account authorised to request a GPU via **PBS**.
- A Python environment on the cluster with `fastapi`, `uvicorn`, and `python-multipart`.
- For eventual live input: the working Aria Gen 2 Python receiver on Megatron.
- **Two or three terminals**: one for the interactive HPC job/server, one for the laptop SSH tunnel, and one for laptop tests/streaming.

Replace `<IMPERIAL_USERNAME>` and `<COMPUTE_NODE>` with your actual values. Compute-node allocations can change for every job.

## 1. Connect to CX3 from Megatron

```bash
ssh <IMPERIAL_USERNAME>@login.cx3.hpc.imperial.ac.uk
```

CX3 uses **PBS**, not Slurm: use `qsub`/`qstat`/`qdel`, **not `srun`**.

## 2. Get an interactive GPU allocation

**On the cluster login node:**

```bash
qsub -I \
  -l select=1:ncpus=4:mem=24gb:ngpus=1 \
  -l walltime=02:00:00
```

The command may wait in the queue. **Do not run your VLM or the persistent FastAPI service on `login-ai`.** When PBS starts your interactive job, the shell should be on an allocated compute node. Verify **before starting the server**:

```bash
hostname
nvidia-smi
printf 'PBS job: %s\n' "$PBS_JOBID"
```

`hostname` should print a node name such as `cx3-4-26`, **not `login-ai`**. The particular node name in this README is from our successful session and is **not a permanent destination**.

## 3. Set up the cluster Python receiver

**On the allocated compute node:**

```bash
# If using the existing environment:
source ~/aria-cluster-env/bin/activate

# Otherwise, create an environment with an available cluster Python module/version.
# python -m venv ~/aria-cluster-env
# source ~/aria-cluster-env/bin/activate

python -m pip install fastapi uvicorn python-multipart
```

In your working directory, create `server.py`:

```python
from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
import time

app = FastAPI()
OUTPUT_DIR = Path.home() / "aria_received_frames"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/frame")
async def receive_frame(
    frame: UploadFile = File(...),
    timestamp: str = Form(...),
):
    image_bytes = await frame.read()
    filename = OUTPUT_DIR / f"rgb_{timestamp}.jpg"
    filename.write_bytes(image_bytes)
    print(f"Received {filename} ({len(image_bytes)} bytes)", flush=True)
    return {
        "status": "received",
        "timestamp": timestamp,
        "server_time_ns": time.time_ns(),
    }
```

**Important: use IPv6 binding for the working CX3 setup:**

```bash
python -m uvicorn server:app --host :: --port 8000
```

Keep this terminal open. `server:app` means “import the `app` object from `server.py`”, so run this from the folder containing `server.py` (or supply `--app-dir /path/to/folder`). If the file doesn't exist, Uvicorn reports `Could not import module "server"`.

### Why `--host ::`, not `0.0.0.0`?

On the tested CX3 setup, the compute-node hostname resolved from the login node to **IPv6**:

```text
cx3-4-26 → 2a0c:5bc0:89:a04:1e34:daff:fe7c:693a
```

`--host 0.0.0.0` bound Uvicorn to IPv4. Connecting to the compute node's IPv6 address then returned **Connection refused**. Restarting Uvicorn with `--host ::` made the IPv6 connection work. IPv6 is the important detail for this particular setup; cluster addressing can differ for future allocations.

## 4. Identify the compute node while the server occupies the terminal

Leave the server running. In **a separate SSH terminal**, connect to the login node and check the PBS job:

```bash
qstat -u "$USER"
qstat -f <PBS_JOB_ID> | grep -E 'exec_host|exec_vnode'
```

Example PBS output:

```text
exec_host = cx3-4-26/32*2
```

**Use only `cx3-4-26` as the hostname.** `/32*2` is PBS allocation notation, **not part of the network hostname**. Use the host assigned to *your current job*, not this example.

To test the server from the **login node**:

```bash
curl -6 -v --connect-timeout 5 http://<COMPUTE_NODE>:8000/health
```

Expected response:

```json
{"status":"ok"}
```

If you run `hostname` or `ss -ltn | grep 8000` in the separate login shell, you'll see **`login-ai`**, not the compute node. That doesn't imply the server has stopped: it is running inside the original interactive GPU job.

## 5. Start the SSH tunnel on Megatron

Open a **new laptop terminal**. Keep your compute-node server running.

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8001:<COMPUTE_NODE>:8000 \
  <IMPERIAL_USERNAME>@login.cx3.hpc.imperial.ac.uk
```

Example with the node allocated in our test:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8001:cx3-4-26:8000 \
  <IMPERIAL_USERNAME>@login.cx3.hpc.imperial.ac.uk
```

Enter your SSH password if prompted. **The terminal then appears to sit idle: that is expected.** `-N` creates a tunnel without starting a remote shell.

Meaning of the two ports:

- `127.0.0.1:8001`: a port on **Megatron** (where the sending script connects).
- `<COMPUTE_NODE>:8000`: the remote FastAPI server running on the **GPU node**.

The cluster side is contacted by SSH from the login node; the server must be reachable from there. Our working setup used the node's IPv6-resolvable hostname.

## 6. Test Megatron → CX3

Open **another terminal on Megatron** while the tunnel remains open.

### Health check

```bash
curl http://127.0.0.1:8001/health
```

Expected:

```json
{"status":"ok"}
```

### Upload a tiny text file

```bash
echo "hello" > hello.txt

curl -X POST http://127.0.0.1:8001/frame \
  -F "frame=@hello.txt" \
  -F "timestamp=test123"
```

Expected JSON contains `"status":"received"` and `"timestamp":"test123"`.

On the **cluster**, the server stores the uploaded bytes at:

```text
~/aria_received_frames/rgb_test123.jpg
```

The file has a `.jpg` suffix because the receiver was designed for images, **but its contents are still text**. Verify from any cluster shell that can access the same home directory:

```bash
cat ~/aria_received_frames/rgb_test123.jpg
# hello
```

**This upload succeeded in our session**, establishing that HTTP POSTs traverse the SSH tunnel and arrive at the compute-node FastAPI service.

## 7. Forward actual Aria frames (next integration step)

On Megatron, keep your existing Aria Python receiver and RGB callback. Send each received frame (ideally via a background worker) to:

```text
POST http://127.0.0.1:8001/frame
```

The working endpoint expects `multipart/form-data` with fields:

- `frame`: file bytes (for live video, encode to JPEG first, e.g. with OpenCV).
- `timestamp`: string representing the frame/arrival timestamp.

Minimal standalone example **for testing with an already-saved JPEG**:

```python
from pathlib import Path
import time
import requests

jpeg = Path("/path/to/saved_aria_image.jpg")
with jpeg.open("rb") as f:
    response = requests.post(
        "http://127.0.0.1:8001/frame",
        files={"frame": (jpeg.name, f, "image/jpeg")},
        data={"timestamp": str(time.time_ns())},
        timeout=10,
    )
response.raise_for_status()
print(response.json())
```

For continuous streaming, use a **bounded latest-frame queue** so slow SSH/network/model inference doesn't create a growing lag. Save local copies only if needed. Keep file writing and JPEG encoding outside the SDK callback where possible. The server currently **saves** files; it does **not** yet run VLM inference.

## 8. Troubleshooting we encountered

| Symptom | Explanation / fix |
|---|---|
| `srun: command not found` | This CX3 environment uses **PBS**. Use `qsub -I` for interactive jobs. |
| `Could not import module "server"` | `server.py` didn't exist in the current directory. Create it, then run Uvicorn from its directory. |
| Tunnel logs `Name of service not known` | The forwarding destination accidentally included PBS notation such as `cx3-4-26/32*2`, or another malformed destination. Use just the hostname. |
| `curl -6` to compute node gives `Connection refused` | Uvicorn was listening on IPv4 `0.0.0.0`; restart it with `--host ::` to listen on IPv6. |
| `curl -4` cannot resolve compute hostname | During our test, CX3 DNS returned only an IPv6 address for the node. Use IPv6 (`curl -6`) and IPv6 Uvicorn binding. |
| `hostname` prints `login-ai` | You're in a **new login-node shell**, not the original interactive GPU job. Find the job's allocated host via `qstat -f`. |
| Tunnel terminal sits idle after login | Normal for `ssh -N`; test with a separate terminal and `curl 127.0.0.1:8001/health`. |
| Local `8001` connection fails | Verify the GPU job and server are running, then the tunnel; check `ss -ltnp | grep ':8001'` on Megatron. |

## 9. Clean shutdown and recreation checklist

1. Stop your laptop's frame-sending Python process.
2. Stop the SSH tunnel on Megatron with **Ctrl+C**.
3. Stop Uvicorn on the GPU node with **Ctrl+C**.
4. Exit the interactive PBS shell; the allocation will end. If needed, cancel a job with `qdel <PBS_JOB_ID>` from the login node.

**To recreate:**

```text
1. SSH to CX3 login node.
2. qsub -I ... ngpus=1 → wait for GPU node allocation.
3. On compute node, activate environment and run:
      python -m uvicorn server:app --host :: --port 8000
4. From another login shell, get the current hostname via qstat -f and test curl -6.
5. On Megatron, establish -L 127.0.0.1:8001:<NODE>:8000 SSH tunnel.
6. On Megatron, curl localhost:8001/health; upload hello.txt as a smoke test.
7. Start existing Aria receiver and POST JPEGs to localhost:8001/frame.
```

## Notes and limits

- This setup is **temporary**: an interactive job ends when its allocation ends, and the compute node may change next time.
- Confirm that using SSH port forwarding and long-lived HTTP services on compute nodes complies with your research group's and Imperial HPC policies.
- The receiving endpoint saves **every** submitted frame; at 5–10 Hz this can generate many files. Disable disk saving or add retention limits when switching to a VLM.
- Nothing here exposes port 8001 publicly on Megatron; it is bound to `127.0.0.1`. The cluster-side FastAPI server is bound to IPv6 `::`, potentially reachable by other cluster hosts: this is suitable only under your cluster's access policy. For production, add authentication/access controls or a more restricted network arrangement.
- The text-upload smoke test doesn't establish real-time throughput, VLM latency, or streaming reliability.

## Reference

[Imperial RCS HPC user guide](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/)
