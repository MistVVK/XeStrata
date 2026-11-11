r"""pytest: setup.py's data-folder migration (#198) saves the data folder it settled on in the user's settings file
(%APPDATA%\Strata\settings.json, ~/.config/strata/settings.json) and can MOVE model files from a folder it finds there.
A test that reaches it must never touch the real file, so the whole session gets its own config home."""
import os
import tempfile

import pytest as _pytest


@_pytest.fixture(autouse=True)
def _fresh_host_memory():
    """The network rule remembers, per process, a host that did not answer (strata_control/netpolicy.py HOSTS). A test run is one process: each
    test starts with an empty memory, so one test's dead fake server does not make the next test's request "fail a moment ago"."""
    def forget():
        try:
            from strata_control import netpolicy
            netpolicy.HOSTS.forget()
        except ImportError:
            pass
    forget()
    yield
    forget()

_home = tempfile.mkdtemp(prefix="strata-pytest-config-")
os.environ["APPDATA"] = _home
os.environ["XDG_CONFIG_HOME"] = _home
# The per-user STATE and DATA folders too (%LOCALAPPDATA%\Strata: tuning/recheck.json, the engine store, the update records; ~/.local/state, ~/.local/share).
# A test that installs a stub engine marks saved tuning presets for a re-check there: on 2026-10-24 one such run left an entry in the owner's real
# tuning\recheck.json. A test that wants a place of its own still names it (STRATA_COMPONENTS_DIR, state_base=...).
os.environ["LOCALAPPDATA"] = _home
os.environ["XDG_STATE_HOME"] = _home
os.environ["XDG_DATA_HOME"] = _home
# Run from PowerShell 7 (GitHub's Windows runners), PSModulePath names PowerShell 7's module folders first: a Windows PowerShell 5.1 child that a
# test starts (install.ps1, fetch-node.ps1) then finds their Microsoft.PowerShell.Utility/Security and cannot load them ("Get-FileHash is not
# recognized"). Keep only the folders that are not PowerShell 7's, as in a Windows PowerShell or cmd window.
if os.name == "nt" and os.environ.get("PSModulePath"):
    _keep = [p for p in os.environ["PSModulePath"].split(os.pathsep)
             if p and not ("\\powershell\\" in p.lower().replace("/", "\\") and "\\windowspowershell\\" not in p.lower().replace("/", "\\"))]
    os.environ["PSModulePath"] = os.pathsep.join(_keep)
# The tests run where GitHub's runners do: no NVIDIA card the code can read. On a box with one (p1004's P100, Boston's A6000s, l620) the
# real nvidia-smi answered and every check that reads the card (engine choice, a stub engine's own start, setup) judged the stand-in
# packages against it: sm_60 has no code in them, so 64 update tests failed on p1004. A `nvidia-smi` that answers nothing and fails
# comes first on PATH for the whole run, child processes included; the code reads that as "cannot tell", as without the tool, and a
# test that needs a card puts its own stand-in first (tools/feed_model_install_gate.py does when `nvidia-smi -L` fails). A test about a
# card patches the reader (engine_component.cuda_archs_present, setup.gpus, ...) as before. POSIX only for now (a stub there would be
# a .cmd, which code that runs `nvidia-smi` directly does not start); the Windows test boxes' cards fit the stand-ins.
if os.name != "nt" and not os.environ.get("STRATA_TEST_REAL_NVIDIA_SMI"):
    _smi = os.path.join(_home, "no-nvidia-smi")
    os.makedirs(_smi, exist_ok=True)
    with open(os.path.join(_smi, "nvidia-smi"), "w") as _f:
        _f.write("#!/bin/sh\n# tests: no NVIDIA card can be read here (conftest.py): it answers nothing and fails, as without a loaded driver\nexit 1\n")
    os.chmod(os.path.join(_smi, "nvidia-smi"), 0o755)
    os.environ["PATH"] = _smi + os.pathsep + os.environ.get("PATH", "")

# No test reaches the network (lane NET, 11-10). A test whose fake was bypassed went to the real hosts (on 11-10 a non-hermetic test pulled
# 1.2 GB), and one on a box without internet hung for the whole timeout. Every connect and every name lookup of this process is checked:
# loopback and local sockets pass; anything else is refused at once (ConnectionRefusedError / gaierror, which the code under test reads as
# "offline") and the test FAILS, naming the host. A live test sets STRATA_TEST_ALLOW_NETWORK=1 itself (the
# STRATA_LIVE_HF test class in tools/ does). Child processes are not covered (the install.sh tests use loopback servers only).
# Three things that are not "the network" pass, named here: a UDP connect() (it only names the peer and sends nothing: tests that ask which address
# leads out), a lookup of this computer's own host name, and the RFC 5737 documentation addresses (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24,
# which nothing answers: a test that needs an address that cannot work); they are called (a), (b) and (c) below. Every use is counted per test and
# the list is printed once at the end of the run (pytest_terminal_summary), so the allowances cannot quietly grow; tests/terminal/test_net_guard.py
# pins that anything else (an address outside them, a real host name, a TCP connect outside the ranges) still fails. The failure message also says
# where in the code the reach came from.
import ipaddress  # noqa: E402
import socket  # noqa: E402

import pytest  # noqa: E402

_reached: list = []
_allowed: list = []                                          # (test id, which allowance, the address): printed once at the end of the run
_current = {"id": "(outside a test)"}


def _allow(which: str, address) -> None:
    _allowed.append((_current["id"], which, str(address)))
_OWN_NAME = socket.gethostname().lower()                     # this computer's own name: a lookup of it is a lookup of itself (tests that read the machine's address)


def _where() -> str:
    """The innermost frame outside this file and the standard library: where the reach came from."""
    import sysconfig
    import traceback
    stdlib = sysconfig.get_paths()["stdlib"].replace("\\", "/")
    for frame in reversed(traceback.extract_stack(limit=40)):
        name = frame.filename.replace("\\", "/")
        if name.endswith("/conftest.py") or name.startswith(stdlib):
            continue
        return f"{name.rsplit('/', 1)[-1]}:{frame.lineno}"
    return "?"


def _network_allowed() -> bool:
    return os.environ.get("STRATA_TEST_ALLOW_NETWORK") == "1"


_DOC_NETS = [ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")]


def _is_documentation(host) -> bool:
    try:
        return any(ipaddress.ip_address(str(host).strip("[]")) in net for net in _DOC_NETS)
    except ValueError:
        return False


def _is_loopback(host) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    host = str(host).strip("[]").split("%", 1)[0].lower()
    if host in ("", "localhost", "localhost.localdomain", "ip6-localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


_real_connect, _real_connect_ex, _real_getaddrinfo = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo


def _check(sock, address) -> None:
    if not isinstance(address, tuple) or getattr(sock, "family", None) not in (socket.AF_INET, socket.AF_INET6):
        return                                              # a local socket (AF_UNIX) or an address form that is not an IP one
    if _is_loopback(address[0]) or _network_allowed():
        return
    if getattr(sock, "type", None) == socket.SOCK_DGRAM:
        _allow("(a) UDP connect", f"{address[0]}:{address[1]}")
        return                                              # a UDP connect() only names the peer: it sends nothing (a test that asks which address leads out)
    if _is_documentation(address[0]):
        _allow("(c) documentation address", f"{address[0]}:{address[1]}")
        return                                              # RFC 5737 TEST-NET: reserved so that nothing answers (a test that needs an address that cannot work)
    _reached.append(f"{address[0]}:{address[1]} [{_where()}]")
    raise ConnectionRefusedError(f"a test reached the network ({address[0]}:{address[1]}): fake it, or set STRATA_TEST_ALLOW_NETWORK=1 in a live test")


def _guarded_connect(self, address):
    _check(self, address)
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    _check(self, address)
    return _real_connect_ex(self, address)


def _guarded_getaddrinfo(host, *args, **kwargs):
    if str(host).lower() == _OWN_NAME and not _is_loopback(host):
        _allow("(b) own host name", host)
    if not (_is_loopback(host) or _network_allowed() or str(host).lower() == _OWN_NAME):
        _reached.append(f"{host} [{_where()}]")
        raise socket.gaierror(socket.EAI_NONAME, f"a test looked up {host}: fake it, or set STRATA_TEST_ALLOW_NETWORK=1 in a live test")
    return _real_getaddrinfo(host, *args, **kwargs)


socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex
socket.getaddrinfo = _guarded_getaddrinfo


@pytest.fixture(autouse=True)
def _no_network(request):
    """Fails the test that reached for a host other than this computer (after it ran, so its own result is not hidden)."""
    del _reached[:]
    _current["id"] = request.node.nodeid
    yield
    _current["id"] = "(outside a test)"
    if _reached:
        hosts = sorted(set(_reached))
        del _reached[:]
        pytest.fail(f"{request.node.nodeid} reached the network: {', '.join(hosts)} (fake it, or set STRATA_TEST_ALLOW_NETWORK=1 in a live test)",
                    pytrace=False)


def pytest_terminal_summary(terminalreporter):
    """The allowances the guard made in this run, once: which test, which allowance, which address."""
    if not _allowed:
        return
    terminalreporter.section("network guard: allowances used")
    for test_id, which, address in sorted(set(_allowed)):
        terminalreporter.write_line(f"{test_id}   {which}   {address}")
