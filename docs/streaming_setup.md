# Project Aria Gen 2 — Wi-Fi hotspot and live streaming (Ubuntu)

This guide documents the setup tested on **Megatron (Ubuntu)** to stream **Project Aria Gen 2** data to a laptop using **`profile9`**, while retaining the laptop's separate Wi-Fi connection for future HPC inference.

## Network topology

```text
Project Aria Gen 2 glasses
         │  Wi-Fi station (`wifi_sta`)
         ▼
Laptop hotspot: AriaStreaming (5 GHz, channel 149)
         │  virtual AP interface: aria_ap
         ▼
Ubuntu laptop (Aria SDK + streaming viewer)
         │  separate Wi-Fi client: wlp0s20f3
         ▼
_The Cloud (Internet; channel 149) → future SSH tunnel → HPC cluster
```

**Important:** The laptop uses **one physical Wi-Fi radio**, with two virtual interfaces. The adapter reports AP + client concurrency but **only on the same channel**. If the upstream Wi-Fi roams to a different channel, the hotspot can stop working. Check your local network's policies before operating a hotspot.

## Prerequisites

- Project Aria Gen 2 glasses and a USB data cable for setup/control.
- Working Aria Gen 2 Client SDK installed in `~/projectaria_gen2_python_env` (adjust path if different).
- Ubuntu with `nmcli`, `iw`, `ip`; `hostapd` for the access point; `dnsmasq` for DHCP.
- A Wi-Fi adapter supporting `managed` + `AP` simultaneously on the same channel (check with `iw list`, under `valid interface combinations`).
- Glasses authenticated with the laptop (`aria_gen2 auth check`, or `aria_gen2 auth pair` if necessary).

Install hotspot dependencies (one time):

```bash
sudo apt update
sudo apt install iw hostapd dnsmasq
```

## One-time hotspot configuration

### 1. Set the upstream Wi-Fi connection to the chosen channel

The working upstream connection was `_The Cloud` on **5 GHz, channel 149 (5745 MHz)**. It uses OWE transition mode. **Restrict the channel, not the BSSID:** pinning the visible transition-mode BSSID caused connection failures.

Find your current profile:

```bash
nmcli -f NAME,UUID,TYPE,DEVICE connection show
```

For this laptop, the active profile was `_The Cloud` (`b17ec864-1af6-443c-8eb2-836b78e86cc1`). Restrict its channel:

```bash
nmcli connection modify "_The Cloud" \
  802-11-wireless.bssid "" \
  802-11-wireless.band a \
  802-11-wireless.channel 149
nmcli connection up "_The Cloud" ifname wlp0s20f3
```

Verify:

```bash
iw dev wlp0s20f3 link
# freq: 5745.0
```

**Caution:** This pins the upstream connection to one channel. If channel 149 is unavailable in another location, restore automatic selection with the recovery command below.

### 2. Create the virtual hotspot interface

If `aria_ap` does not already exist:

```bash
sudo iw dev wlp0s20f3 interface add aria_ap type __ap
```

NetworkManager may reset it to client (`managed`) mode or mark it `unavailable`. The working workaround was to let **hostapd**, not NetworkManager, manage the hotspot:

```bash
sudo nmcli device set aria_ap managed no
sudo ip link set aria_ap down
sudo iw dev aria_ap set type __ap
sudo ip link set aria_ap up
```

Check:

```bash
iw dev aria_ap info
# type AP
```

If the interface is already in AP mode, do not recreate it. Virtual interfaces may need to be recreated after reboot.

### 3. Create the hostapd configuration

Create `/etc/hostapd/aria.conf` (one time):

```bash
sudo mkdir -p /etc/hostapd
sudo nano /etc/hostapd/aria.conf
```

Contents:

```ini
interface=aria_ap
driver=nl80211
ssid=AriaStreaming
hw_mode=a
channel=149
country_code=GB
ieee80211d=1
wpa=2
wpa_passphrase=CHANGE_ME_TO_A_STRONG_PASSWORD
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
```

Replace the example password with your own **8–63-character** WPA2 passphrase. The glasses must use the same password.

### 4. Configure a local IP address and DHCP

The AP needs both an IP address and a DHCP server so the glasses can join. We used the local subnet `192.168.50.0/24`.

Assign the laptop's hotspot IP:

```bash
sudo ip addr replace 192.168.50.1/24 dev aria_ap
```

Create `/etc/dnsmasq-aria.conf`:

```ini
interface=aria_ap
bind-interfaces
dhcp-range=192.168.50.10,192.168.50.100,255.255.255.0,12h
dhcp-option=option:router,192.168.50.1
```

No internet sharing/NAT is required for glasses-to-laptop streaming. The separate upstream Wi-Fi is used by the laptop itself. If a system-wide dnsmasq service conflicts with the dedicated instance, inspect the port conflict before disabling anything.

## Starting a streaming session

Use separate terminal windows; keep the hotspot and DHCP processes running while streaming.

### Terminal A — ensure AP interface and start the Wi-Fi hotspot

```bash
# Confirm upstream is still on channel 149
iw dev wlp0s20f3 link

sudo nmcli device set aria_ap managed no
sudo ip link set aria_ap down
sudo iw dev aria_ap set type __ap
sudo ip link set aria_ap up

sudo hostapd /etc/hostapd/aria.conf
```

Look for **`aria_ap: AP-ENABLED`**. `hostapd -dd` is useful for debugging but prints continuous probe-request and probe-response logs; this is normal, so use `hostapd` without `-dd` for regular sessions.

### Terminal B — start DHCP

```bash
sudo ip addr replace 192.168.50.1/24 dev aria_ap
sudo dnsmasq --no-daemon --conf-file=/etc/dnsmasq-aria.conf --port=0
```

A phone connected to `AriaStreaming` should receive a `192.168.50.x` address. It may say **No Internet**, which is expected. This test verifies Wi-Fi association and DHCP, not internet sharing.

### Terminal C — connect the glasses and start `profile9`

Connect the glasses by USB for setup/control, then:

```bash
source ~/projectaria_gen2_python_env/bin/activate
aria_gen2 device list
aria_gen2 auth check
aria_gen2 device wifi connect --help  # Check flags for your installed SDK
```

Configure the glasses for the hotspot (replace the password):

```bash
aria_gen2 device wifi connect \
  --ssid "AriaStreaming" \
  --password "YOUR_HOTSPOT_PASSWORD" \
  --skip-internet-check
```

The `--skip-internet-check` flag is useful because the hotspot is local-only. If your installed SDK rejects it, inspect `aria_gen2 device wifi connect --help` and use the supported equivalent.

Start the wireless stream:

```bash
aria_gen2 streaming start \
  --profile profile9 \
  --interface wifi_sta \
  --batch-period-ms 200
```

After streaming starts successfully, the USB cable can be unplugged for wireless operation, subject to the device/SDK requirements.

### Terminal D — view live data

```bash
source ~/projectaria_gen2_python_env/bin/activate
aria_streaming_viewer --real-time --interpolate
```

This opens the **Rerun** viewer. In the tested setup it successfully displayed the live Aria stream. `profile9` includes **RGB at 5 Hz**, along with additional sensor and machine-perception outputs; it is **not RGB-only**.

## Stop and clean up

1. Close the viewer (`Ctrl+C` in its terminal).
2. Stop the Aria streaming session (reconnect USB if needed for control):

   ```bash
   aria_gen2 streaming stop
   ```

3. Stop `hostapd` and the foreground `dnsmasq` processes (`Ctrl+C` in their terminals).
4. Optionally remove the hotspot IP:

   ```bash
   sudo ip addr del 192.168.50.1/24 dev aria_ap
   ```

5. Optionally restore your Wi-Fi profile to automatic channel selection:

   ```bash
   nmcli connection modify "_The Cloud" \
     802-11-wireless.bssid "" \
     802-11-wireless.band "" \
     802-11-wireless.channel 0
   nmcli connection up "_The Cloud" ifname wlp0s20f3
   ```

6. Optionally delete the temporary virtual interface (after stopping hostapd):

   ```bash
   sudo iw dev aria_ap del
   ```

## Troubleshooting

| Symptom | What helped / what to check |
|---|---|
| `Failed to connect to streaming network` | Confirm glasses joined `AriaStreaming`, DHCP assigned an IP, and `hostapd` is active; keep USB connected while initiating the session. |
| `band argument value '6GHz' is invalid` | The installed `nmcli` hotspot command accepted only `a`/`bg`. Use the working 5 GHz path instead. |
| `_The Cloud` connection fails when pinned to BSSID | Network uses **OWE transition mode**. Pin the **channel** instead of a BSSID; that worked on this machine. |
| `aria_ap` is `unavailable` in `nmcli` | `nmcli` was not able to launch the virtual AP. Set `aria_ap` unmanaged and use `hostapd`. |
| `RTNETLINK answers: Device or resource busy` | After setting `aria_ap` unmanaged, bringing it down, setting `type __ap`, then bringing it up succeeded. |
| `hostapd` prints endless hex/probe logs | You started it with `-dd`; use `sudo hostapd /etc/hostapd/aria.conf` instead. |
| `Could not open configuration file` | Create `/etc/hostapd/aria.conf` and verify its path and permissions. |
| Hotspot visible but glasses cannot use it | Check password, `hostapd` association logs, `dnsmasq` DHCP leases, firewall, and subnet clashes. |
| Internet/HPC connection disappears | Check `iw dev wlp0s20f3 link`: upstream must stay on the same channel as the hotspot. Pinning can weaken coverage or prevent roaming. |

Useful inspection commands:

```bash
nmcli device status
iw dev
iw dev wlp0s20f3 link
ip -4 addr show aria_ap
sudo journalctl -b -u NetworkManager --no-pager | tail -80
```

## Next step: VLM on university HPC

The demonstrated path so far is **Aria → hotspot → laptop → live Rerun viewer**. The HPC link is not yet implemented. Next, use the Aria Python SDK to receive RGB images, retain only the latest frame, and transmit selected frames through an **approved SSH tunnel** to a VLM server running inside a scheduled GPU job. Keep image capture asynchronous from inference to avoid building a stale-frame queue.

## References

- [Project Aria Gen 2 Client SDK — Getting started](https://facebookresearch.github.io/projectaria_tools/gen2/ark/client-sdk/start)
- [Project Aria Gen 2 Client SDK — Streaming](https://facebookresearch.github.io/projectaria_tools/gen2/ark/client-sdk/streaming)
- [Project Aria Gen 2 device profiles](https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/device/profile)

