"""Create vector closeout figures from verified metrics; no simulated data."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"figures"; OUT.mkdir(exist_ok=True)
m=json.loads((ROOT/"metadata/verified_metrics.json").read_text())
NAVY="#17324D"; TEAL="#087F8C"; GOLD="#B87514"; GRAY="#62758A"; LIGHT="#EDF3F7"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":17,"pdf.fonttype":42,"svg.fonttype":"none","text.color":NAVY,"axes.labelcolor":NAVY,"xtick.color":GRAY,"ytick.color":NAVY})
def save(fig,name):
    fig.savefig(OUT/(name+".pdf"),bbox_inches="tight",pad_inches=.12,facecolor="white")
    fig.savefig(OUT/(name+".png"),dpi=150,bbox_inches="tight",pad_inches=.12,facecolor="white")
    plt.close(fig)
def box(ax,x,y,w,h,title,body,accent=TEAL,fs=16):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.008,rounding_size=0.018",lw=1.4,ec=accent,fc=LIGHT))
    ax.text(x+w/2,y+h*.72,title,ha="center",va="center",weight="bold",fontsize=fs,color=accent)
    ax.text(x+w/2,y+h*.32,body,ha="center",va="center",fontsize=fs-1,linespacing=1.3)
def arrow(ax,p,q,color=GRAY):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=17,lw=1.7,color=color))
fig,ax=plt.subplots(figsize=(8.4,6.7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
box(ax,.2,.84,.6,.135,"STATE + COMMANDS","body state / wheel-leg feedback")
box(ax,.035,.565,.425,.18,"CLASSICAL STACK","LQR + calibrations\nposture reference",fs=16)
box(ax,.54,.565,.425,.18,"BOUNDED RESIDUAL","PPO / fixed 6-D interface\nstage-specific authority",fs=16)
arrow(ax,(.39,.832),(.25,.754));arrow(ax,(.61,.832),(.75,.754))
box(ax,.16,.315,.68,.15,"COMPOSE + LIMIT","mask / scale / clip / slew")
arrow(ax,(.25,.556),(.4,.475));arrow(ax,(.75,.556),(.6,.475))
box(ax,.16,.075,.68,.145,"MJLAB / MUJOCO","physics contract fixed per experiment")
arrow(ax,(.5,.305),(.5,.23))
ax.plot([.845,.99,.99,.81],[.145,.145,.907,.907],color=GRAY,lw=1.5)
arrow(ax,(.9,.907),(.81,.907))
ax.text(.03,.022,"Zero policy output = the complete classical baseline",fontsize=14,color=GRAY)
save(fig,"control_architecture")
fig,ax=plt.subplots(figsize=(8.4,6.5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
steps=[("01  PROBE + CALIBRATE","measure the operating envelope"),("02  FREEZE THE CONTRACT","revision / artifact hashes / limits"),("03  TRAIN WITH BOUNDS","capability mask / migration checks"),("04  EVALUATE","rejection screen, then formal comparison"),("05  RECORD THE DECISION","pass, valid negative, or diagnostic")]
for i,(title,body) in enumerate(steps):
    y=.825-i*.185
    box(ax,.035,y,.925,.145,title,body,fs=17)
    if i<4:arrow(ax,(.5,y-.009),(.5,y-.040))
ax.text(.5,.025,"A completed run is not automatically a passed capability.",ha="center",fontsize=14,color=GOLD)
save(fig,"evidence_workflow")
r,a=m["recovery"]
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8.5,7.0));fig.subplots_adjust(hspace=.65,left=.27,right=.92,top=.92,bottom=.1)
ax1.barh([1,0],[r["baseline_s"],r["candidate_s"]],color=[GRAY,TEAL],height=.48)
ax1.set_yticks([1,0],["Classical","Hybrid"]);ax1.set_xlim(0,1.18);ax1.set_xlabel("Recovery time (s) - lower is better",fontsize=16)
ax1.set_title("A  Matched recovery comparison",loc="left",weight="bold",fontsize=19,pad=15)
for y,x in [(1,r["baseline_s"]),(0,r["candidate_s"])]:ax1.text(x+.02,y,f"{x:.4f}",va="center",fontsize=18,weight="bold")
ax2.barh([1,0],[r["improvement_pct"],a["improvement_pct"]],color=[TEAL,GOLD],height=.48)
ax2.set_yticks([1,0],["Hybrid","Legs masked"]);ax2.set_xlim(0,15);ax2.set_xticks([0,5,10,15]);ax2.set_xlabel("Improvement vs. own matched baseline (%)",fontsize=15)
ax2.set_title("B  Evaluation-time leg ablation",loc="left",weight="bold",fontsize=19,pad=20)
ax2.axvline(10,color=NAVY,ls="--",lw=1.3,zorder=0)
ax2.text(10.15,.4,"10% gate",fontsize=13,color=NAVY)
for y,x in [(1,r["improvement_pct"]),(0,a["improvement_pct"])]:ax2.text(x+.22,y,f"{x:.2f}%",va="center",fontsize=18,weight="bold")
for ax in (ax1,ax2):
    ax.spines[["top","right","left"]].set_visible(False);ax.spines["bottom"].set_color("#C7D3DD");ax.tick_params(axis="y",length=0,pad=12);ax.grid(axis="x",alpha=.16,zorder=0);ax.set_axisbelow(True)
save(fig,"representative_results")
print("Wrote three vector figures and PNG previews.")
