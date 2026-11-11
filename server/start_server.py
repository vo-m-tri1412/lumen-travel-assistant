"""Idempotent Colab setup job. Run in the background so the kernel stays free."""
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import time
from urllib.error import URLError,HTTPError
from urllib.request import Request,urlopen

from .tunnel import select_tunnel_url

ROOT=Path(__file__).resolve().parents[1]
STATE=Path("/tmp/lumen-server.json")
LOG=Path("/tmp/lumen-setup.log")
HOST="http://127.0.0.1:11434"
state={"stage":"starting","ready":False,"pid":os.getpid()}


def save(stage,**values):
    state.update(stage=stage,**values)
    temp=STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state),encoding="utf-8")
    temp.replace(STATE)


def get_json(url,headers=None,timeout=5):
    with urlopen(Request(url,headers=headers or {}),timeout=timeout) as response:
        return json.loads(response.read())


def run(command,log,timeout=300,env=None):
    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,env=env)
    if result.returncode:raise RuntimeError(f"Command failed with exit {result.returncode}; inspect {LOG}.")


def stop_owned_processes():
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args=path.read_bytes().decode(errors="replace").strip("\0").split("\0")
            proxy=str(ROOT/"server/proxy.py") in args
            tunnel=bool(args) and Path(args[0]).name=="ssh" and "0:localhost:11435" in args and any(host in args for host in ["free.pinggy.io","a.pinggy.io"])
            if (proxy or tunnel) and int(path.parent.name)!=os.getpid():os.kill(int(path.parent.name),signal.SIGTERM)
        except (OSError,ValueError):pass


def main():
    # Linux/Colab file lock prevents duplicate setup jobs after notebook reruns.
    import fcntl
    lock=Path("/tmp/lumen-setup.lock").open("w")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print("A Lumen setup job is already running.",flush=True)
        return
    with LOG.open("a",encoding="utf-8",buffering=1) as log:
        try:
            gpu=subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],capture_output=True,text=True)
            if gpu.returncode:raise RuntimeError("Select a Colab GPU before starting setup.")
            state["gpu"]=gpu.stdout.strip()
            if STATE.exists():
                previous=json.loads(STATE.read_text())
                if previous.get("ready"):
                    try:
                        headers={"Authorization":"Bearer "+previous["api_token"],"bypass-tunnel-reminder":"true"}
                        get_json(previous["url"]+"/api/tags",headers,10)
                        print("Existing Lumen server is ready.",flush=True)
                        return
                    except Exception:pass
            save("prerequisites")
            if any(shutil.which(tool) is None for tool in ["curl","zstd","lspci","lshw","ssh-keygen"]):
                run(["apt-get","update","-qq"],log)
                run(["apt-get","install","-y","-qq","curl","ca-certificates","zstd","pciutils","lshw","openssh-client"],log)
            if not shutil.which("ollama"):
                save("installing_ollama")
                run(["bash","-lc","set -o pipefail; curl --retry 3 -fsSL https://ollama.com/install.sh | sh"],log)
            env={**os.environ,"OLLAMA_HOST":"127.0.0.1:11434","OLLAMA_MAX_LOADED_MODELS":"1","OLLAMA_NUM_PARALLEL":"1"}
            save("starting_ollama")
            try:get_json(HOST+"/api/tags")
            except Exception:
                ollama_log=Path("/tmp/lumen-ollama.log").open("a")
                subprocess.Popen(["ollama","serve"],env=env,stdout=ollama_log,stderr=subprocess.STDOUT,start_new_session=True)
                for _ in range(40):
                    try:get_json(HOST+"/api/tags");break
                    except Exception:time.sleep(1)
                else:raise RuntimeError("Ollama did not start within 40 seconds.")
            for model in ["embeddinggemma","qwen3:4b"]:
                save("pulling_"+model)
                run(["ollama","pull",model],log,600,env)
            stop_owned_processes()
            token=secrets.token_urlsafe(32)
            save("starting_proxy")
            proxy_log=Path("/tmp/lumen-proxy.log").open("a")
            proxy=subprocess.Popen([sys.executable,str(ROOT/"server/proxy.py")],env={**os.environ,"OLLAMA_API_TOKEN":token},stdout=proxy_log,stderr=subprocess.STDOUT,start_new_session=True)
            time.sleep(1)
            if proxy.poll() is not None:raise RuntimeError("Proxy did not start; inspect /tmp/lumen-proxy.log.")
            key=Path("/tmp/lumen-pinggy-key")
            if not key.exists():run(["ssh-keygen","-q","-t","ed25519","-N","","-f",str(key)],log)
            save("starting_pinggy")
            tunnel_log=Path("/tmp/lumen-tunnel.log").open("a")
            tunnel=subprocess.Popen(["ssh","-T","-p","443","-i",str(key),"-o","IdentitiesOnly=yes","-o","BatchMode=yes","-o","StrictHostKeyChecking=accept-new","-o","ServerAliveInterval=30","-o","ExitOnForwardFailure=yes","-R","0:localhost:11435","-L","127.0.0.1:4300:localhost:4300","free.pinggy.io"],stdin=subprocess.DEVNULL,stdout=tunnel_log,stderr=subprocess.STDOUT,start_new_session=True)
            url=None
            # Use Pinggy's documented machine-readable API, not terminal banners.
            deadline=time.monotonic()+40
            while time.monotonic()<deadline:
                if tunnel.poll() is not None:raise RuntimeError("Pinggy exited; inspect /tmp/lumen-tunnel.log.")
                try:
                    url=select_tunnel_url(get_json("http://127.0.0.1:4300/urls",timeout=2))
                    if url:break
                except Exception:pass
                time.sleep(1)
            if not url:raise RuntimeError("Pinggy did not return an HTTPS URL within 40 seconds.")
            headers={"Authorization":"Bearer "+token,"bypass-tunnel-reminder":"true"}
            models=get_json(url+"/api/tags",headers,30)["models"]
            save("ready",ready=True,url=url,api_token=token,models=[m["name"] for m in models],proxy_pid=proxy.pid,tunnel_pid=tunnel.pid)
            print("Lumen server is ready. Connection values are in /tmp/lumen-server.json.",flush=True)
        except Exception as error:
            save("error",ready=False,error=str(error),log=str(LOG))
            print(f"Setup failed: {error}",file=log,flush=True)
            raise


if __name__=="__main__":main()
