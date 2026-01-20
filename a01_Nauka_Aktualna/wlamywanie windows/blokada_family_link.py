import ctypes as c
import getpass as g
import os
import subprocess as s
import sys
import time as t

oa=os.path.abspath;sh=c.windll.shell32;ex=sys.executable
class S:
 def __init__(_):
  _.u=g.getuser();y=r"C:\Windows\System32";_.f=0x08000000;_.s="WpcMonSvc"
  _.t=[f for f in os.listdir(y) if f.lower().startswith("wpc") and f.endswith(".exe")] if os.path.exists(y) else["WpcMon.exe","WpcTok.exe"]
 def a(_):sh.IsUserAnAdmin()or(sh.ShellExecuteW(0,"runas",ex,f'"{oa(__file__)}"',0,1),sys.exit());return _
 def d(_):
  r=s.run;[r(f'netsh advfirewall firewall add rule name=X_{e}_{d} dir={d} action=block program=C:\\Windows\\System32\\{e}&&taskkill /F /IM {e} /T',shell=1,creationflags=_.f)for e in _.t for d in['in','out']]
  r(f'net user {_.u} /time:all&sc config {_.s} start= disabled&sc stop {_.s}',shell=1,creationflags=_.f)
 def m(_,h=0):
  if 1-h:
   os.system(f"title T - {_.u}");print(f"[*] {_.u}")
   if input("G? [t/n]: ")=="t":s.Popen([ex.replace("python.exe","pythonw.exe"),oa(__file__),"g"],creationflags=_.f);sys.exit()
  while 1:_.d();(1-h)and(print(f"[{t.strftime('%H:%M:%S')}] OK",end="\r"),t.sleep(5))
if __name__=="__main__":S().a().m(h=len(sys.argv)>1)
