/**
 * Windows graphics cards that nvidia-smi does not cover (AMD, Intel): their size and their load.
 *  - size: the DirectX adapter list in the registry (HKLM\SOFTWARE\Microsoft\DirectX) has a 64-bit DedicatedVideoMemory and the adapter
 *    LUID. Win32_VideoController.AdapterRAM is 32 bits and reads 0 or 4 GB for a 16 GB card (#3727, #3836).
 *  - load and memory in use: the Windows GPU performance counters (the ones Task Manager shows), read by one long-lived PowerShell and
 *    matched to the card by LUID.
 *  - temperature: the display driver's own D3DKMT adapter performance data (gdi32.dll, no vendor library, no administrator), asked for the
 *    same LUID. An AMD 7900 XTX gave the same value as ADLX at idle and under load (noctrex, #3727); the card answers in tenths of a degree.
 *    A driver that does not answer leaves the temperature "not available". Power, fan and the PCIe link stay "not available": D3DKMT gave
 *    0 for power and for the PCIe rate on that card.
 */
import { createPowerShellLoop, endWithHost } from './windows-poll.mjs';
import { utf8Script } from './powershell-utf8.mjs';

export const DX_ADAPTERS_SCRIPT = utf8Script(
  "Get-ChildItem HKLM:\\SOFTWARE\\Microsoft\\DirectX | ForEach-Object { $p = Get-ItemProperty $_.PSPath; " +
  "if ($p.Description -and $null -ne $p.AdapterLuid) { [pscustomobject]@{ Description = [string]$p.Description; VendorId = [string]$p.VendorId; " +
  "Luid = [string]$p.AdapterLuid; Dedicated = [string]$p.DedicatedVideoMemory; Software = [string]$p.SoftwareAdapter } } } | ConvertTo-Json -Compress");

/** luid_0x<high>_0x<low> in the spelling of the performance counters, from the registry's 64-bit number. */
export function luidName(value) {
  try {
    const n = BigInt.asUintN(64, BigInt(String(value)));
    const hex = v => '0x' + v.toString(16).padStart(8, '0');
    return `luid_${hex(n >> 32n)}_${hex(n & 0xffffffffn)}`;
  } catch { return null; }
}

/** The adapters the DirectX registry lists: [{ description, vendorId, luid, bytes, software }]. Never throws. */
export function parseDxAdapters(raw) {
  try {
    const parsed = JSON.parse(raw), rows = Array.isArray(parsed) ? parsed : [parsed];
    return rows.filter(r => r?.Description).map(r => ({
      description: String(r.Description), vendorId: Number(r.VendorId) || 0, luid: luidName(r.Luid),
      bytes: Number(r.Dedicated) || 0, software: String(r.Software) === '1' || String(r.Software).toLowerCase() === 'true',
    }));
  } catch { return []; }
}

const norm = text => String(text ?? '').toLowerCase().replace(/[^a-z0-9]/g, '');
const VIRTUAL = /virtual|remote|indirect|mirror|basic render|meta |parsec|displaylink|idd/i;

/**
 * Win32_VideoController rows + the DirectX adapter list -> the cards this app can show. A card keeps the size from the registry;
 * a display driver for a virtual monitor (no size, no known maker) is not a graphics card and is dropped.
 */
export function windowsOtherGpus(videoRows, dx) {
  const used = new Set();
  const out = [];
  for (const row of videoRows) {
    if (!row?.Name) continue;
    const match = dx.find((a, i) => !used.has(i) && !a.software && norm(a.description) === norm(row.Name));
    if (match) used.add(dx.indexOf(match));
    const vendor = /amd|radeon/i.test(row.Name) ? 'amd' : /intel/i.test(row.Name) ? 'intel' : 'unknown';
    const vramGb = match?.bytes ? Math.round(match.bytes / 1024 ** 3 * 10) / 10 : 0;
    if (vendor === 'unknown' && (!vramGb || VIRTUAL.test(row.Name))) continue;
    out.push({ id: String(row.PNPDeviceID), vendor, name: row.Name, vramGb, luid: match?.luid ?? null });
  }
  return out;
}

/** The temperature D3DKMT reports in tenths of a degree -> whole degrees, or null when the driver gave nothing believable. */
export function d3dkmtTemperatureC(deci) {
  const c = Number(deci) / 10;
  return Number.isFinite(c) && c > 0 && c <= 150 ? Math.round(c) : null;
}

/** One counter line: "G <luid> <load percent> <dedicated bytes in use> [<temperature in tenths of a degree>]". */
export function parseGpuCounterLine(line) {
  const m = /^G\s+(luid_0x[0-9a-f]+_0x[0-9a-f]+)\s+(\d+(?:\.\d+)?)\s+(\d+)(?:\s+(-?\d+))?\s*$/i.exec(String(line));
  return m ? { luid: m[1].toLowerCase(), utilizationPct: Math.max(0, Math.min(100, Number(m[2]))), memoryUsedGb: Number(m[3]) / 1024 ** 3, temperatureC: d3dkmtTemperatureC(m[4]) } : null;
}

/** The C# that asks D3DKMT (KMTQAITYPE_ADAPTERPERFDATA = 62, 64-byte D3DKMT_ADAPTER_PERFDATA, Temperature at byte 56) for one LUID; -1 = no answer. */
export const GPU_TEMP_CSHARP = 'using System;using System.Runtime.InteropServices;public static class StrataGpuTemp{[StructLayout(LayoutKind.Sequential)]struct Q{public uint L;public int H;public uint A;}[StructLayout(LayoutKind.Sequential)]struct Info{public uint A;public int T;public IntPtr P;public uint S;}[DllImport("gdi32.dll")]static extern int D3DKMTOpenAdapterFromLuid(ref Q q);[DllImport("gdi32.dll")]static extern int D3DKMTCloseAdapter(ref uint a);[DllImport("gdi32.dll")]static extern int D3DKMTQueryAdapterInfo(ref Info i);public static int Read(int high,uint low){Q q=new Q();q.L=low;q.H=high;if(D3DKMTOpenAdapterFromLuid(ref q)!=0)return -1;IntPtr buf=Marshal.AllocHGlobal(64);try{for(int k=0;k<64;k++)Marshal.WriteByte(buf,k,0);Info i=new Info();i.A=q.A;i.T=62;i.P=buf;i.S=64;if(D3DKMTQueryAdapterInfo(ref i)!=0)return -1;return Marshal.ReadInt32(buf,56);}finally{Marshal.FreeHGlobal(buf);uint a=q.A;D3DKMTCloseAdapter(ref a);}}}';

export function createWindowsGpuReader({ spawnFn, intervalSec = 4, maxStarts = 3, now = Date.now } = {}) {
  const readings = new Map();
  const script = utf8Script(
    `$t=$false;try{Add-Type -TypeDefinition '${GPU_TEMP_CSHARP}';$t=$true}catch{};` +
    `while($true){` +
    `$e=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine);$m=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory);` +
    `$u=@{};foreach($r in $e){if($r.Name -match '(luid_0x[0-9A-Fa-f]+_0x[0-9A-Fa-f]+)_phys_\\d+_eng_\\d+_engtype_(\\w+)'){$k=$matches[1].ToLower()+'|'+$matches[2];$u[$k]=[double]$u[$k]+[double]$r.UtilizationPercentage}};` +
    `$b=@{};foreach($k in $u.Keys){$l=$k.Split('|')[0];if(-not $b.ContainsKey($l) -or $u[$k] -gt $b[$l]){$b[$l]=$u[$k]}};` +
    `foreach($r in $m){if($r.Name -match '(luid_0x[0-9A-Fa-f]+_0x[0-9A-Fa-f]+)'){$l=$matches[1].ToLower();$d=-1;if($t){try{$p=$l.Split('_');$d=[StrataGpuTemp]::Read([Convert]::ToInt32($p[1].Substring(2),16),[Convert]::ToUInt32($p[2].Substring(2),16))}catch{}};[Console]::Out.WriteLine('G '+$l+' '+[math]::Round([double]$b[$l],0)+' '+$r.DedicatedUsage+' '+$d)}};` +
    `[Console]::Out.Flush();Start-Sleep -Seconds ${intervalSec}}`);
  const loop = createPowerShellLoop({ script, spawnFn, maxStarts, onLine: line => {
    const parsed = parseGpuCounterLine(line);
    if (parsed) readings.set(parsed.luid, { ...parsed, at: now() });
    return !!parsed;
  } });
  return {
    read(luid) { loop.start(); const r = readings.get(String(luid ?? '').toLowerCase()); return r && now() - r.at < 6 * intervalSec * 1000 ? r : null; },
    stop() { loop.stop(); },
  };
}

let shared = null;
/** The card's load and memory in use from the Windows counters; null when there is no reading. */
export function windowsGpuReading(luid) {
  if (process.platform !== 'win32' || !luid) return null;
  shared ??= endWithHost(createWindowsGpuReader());
  return shared.read(luid);
}
