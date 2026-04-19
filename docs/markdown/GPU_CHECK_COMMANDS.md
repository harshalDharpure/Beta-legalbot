# GPU Usage Check Commands (Server)

Use these commands on the server to check GPU availability and who is using them. No need to ask the AI—copy-paste and run.

---

## 1. Quick summary: which GPUs are free and how much memory

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits
```

**Output:** One line per GPU: `index, name, used_MiB, total_MiB, free_MiB, util_%`  
**Free GPU:** Look for high `free_MiB` (e.g. 40000+) and low `used_MiB` (e.g. &lt;1000). Low `util_%` means idle.

---

## 2. List all processes using GPU and how much VRAM

```bash
nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader
```

**Output:** `pid, memory_MiB, gpu_uuid` for each process.  
To map UUID → GPU index, use the next command.

---

## 3. Processes per GPU (with GPU index)

```bash
nvidia-smi -q -d PIDS
```

**Output:** For each GPU (0, 1, 2, …), lists Process ID and Used GPU Memory.  
Scroll to see which PIDs are on which GPU.  
**Shorter (first 80 lines):**

```bash
nvidia-smi -q -d PIDS | head -80
```

---

## 4. Who is the user for each PID?

After you have PIDs from the commands above, get username and command:

```bash
# Replace PIDS with space-separated list, e.g. 3477 12345 67890
for pid in PIDS; do u=$(ps -o user= -p $pid 2>/dev/null); c=$(ps -o comm= -p $pid 2>/dev/null); echo "PID $pid | $u | $c"; done
```

**Example (multiple PIDs):**

```bash
for pid in 3477 2230884 2329711; do u=$(ps -o user= -p $pid 2>/dev/null); c=$(ps -o comm= -p $pid 2>/dev/null); echo "PID $pid | $u | $c"; done
```

---

## 5. One-liner: all GPU PIDs with user and command

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | while read pid; do u=$(ps -o user= -p $pid 2>/dev/null); c=$(ps -o comm= -p $pid 2>/dev/null); echo "PID $pid | $u | $c"; done
```

**Output:** One line per unique PID: `PID 12345 | username | command`

---

## 6. List only MY processes (replace with your username)

```bash
ps -u YOUR_USERNAME -o pid,%cpu,%mem,etime,cmd --no-headers
```

**Example:**

```bash
ps -u vaneet_2221cs15 -o pid,%cpu,%mem,etime,cmd --no-headers
```

---

## 7. Check if a specific training script is running

```bash
ps aux | grep -E "run_new_models_training|train_generation_template" | grep -v grep
```

**Empty output** = no such process running.

---

## 8. GPU mapping (UUID → index)

GPU order in `nvidia-smi -q` is usually:

- `GPU 00000000:01:00.0` → **GPU 0**
- `GPU 00000000:41:00.0` → **GPU 1**
- `GPU 00000000:81:00.0` → **GPU 2**
- `GPU 00000000:C1:00.0` → **GPU 3**
- `GPU 00000000:E1:00.0` → **GPU 4**

---

## 9. Recommended workflow

1. Run **command 1** to see which GPUs have free memory.
2. Run **command 3** (or `nvidia-smi -q -d PIDS`) to see PIDs per GPU.
3. Run **command 4** with those PIDs to see who is using each GPU.
4. Pick a GPU with enough free VRAM and set it when launching your job, e.g.:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python3 models/run_new_models_training.py
   ```

---

*File: `GPU_CHECK_COMMANDS.md` in the legal-bot repo root. Update if your server’s GPU layout or usernames differ.*
