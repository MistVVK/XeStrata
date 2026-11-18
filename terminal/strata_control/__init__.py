"""Local operator contracts shared by terminal and future Desktop clients."""
import sys
if sys.platform == "win32":
    # CPython 3.12's WMI helper (_wmi, behind platform.system() / uname()) takes an access violation in a worker thread when WMI refuses the
    # account (0x80041003) and the caller has already gone (DIAG-WINHOST-0216). Without _wmi, platform uses the registry and the environment.
    sys.modules["_wmi"] = None
