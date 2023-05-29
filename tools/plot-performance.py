#!/usr/bin/env python

import os, sys, re
import numpy as np
import pandas as pd
import matplotlib
# Unless this is set, the script may freeze while it attempts to use X windows
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

found_input = False
interactive = True
for fnm in sys.argv[1:]:
    if fnm.endswith(".in"):
        print("Input file (for this script) detected - will parse it for log filenames.")
        found_input = True
        interactive = False
        break

if found_input:
    logfnms = ["%s/run.log" % line.strip() for line in open(fnm).readlines() if len(line.strip()) > 0]
else:
    logfnms = sys.argv[1:]

print(logfnms)

if len(logfnms) == 0 or any([not os.path.exists(fnm) for fnm in logfnms]):
    print("Exiting because did not provide valid log filenames.")
    print("Usage: ./plot-performance.py <geomeTRIC log filenames>")
    sys.exit()

dirname_default = os.path.split(os.getcwd())[1]
if dirname_default == "saved":
    dirname_default = os.path.split(os.path.split(os.getcwd())[0])[1]
    
fig_title_default = "Optimization performance for %s" % dirname_default
if interactive:
    fig_title = input('Enter custom figure title (Default: %s) -->' % fig_title_default)
    if not fig_title:
        fig_title = fig_title_default
else:
    fig_title = fig_title_default

if interactive:
    labels = input('Enter optional legend labels separated by space (%i required) --> ' % len(logfnms)).split()
else:
    labels = []
if len(labels) != len(logfnms):
    if interactive: print("Did not provide valid labels - using log filenames.")
    labels = logfnms
    # Trim off common prefix and suffix from labels
    for i in range(min([len(l) for l in labels])):
        if len(set([l[i] for l in labels])) != 1: break
    labels = [l[i:][::-1] for l in labels]
    for i in range(min([len(l) for l in labels])):
        if len(set([l[i] for l in labels])) != 1: break
    labels = [l[i:][::-1] for l in labels]

df_energy = pd.DataFrame(dict([(label, []) for label in labels]))
df_grms = pd.DataFrame(dict([(label, []) for label in labels]))
df_gmax = pd.DataFrame(dict([(label, []) for label in labels]))
df_drms = pd.DataFrame(dict([(label, []) for label in labels]))
df_dmax = pd.DataFrame(dict([(label, []) for label in labels]))
df_de = pd.DataFrame(dict([(label, []) for label in labels]))
status = {}

for ifnm, fnm in enumerate(logfnms):
    label = labels[ifnm]
    step = 0
    status[label] = "Unknown"
    for line in open(fnm):
        # Strip ANSI coloring
        line = re.sub(r'\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)*[m|K]','',line)
        if re.match('^Step +[0-9]+ :', line): # Line contains geom-opt. data
            if 'Gradient' in line: # This is the zero-th step
                grms_gmax_word = re.split('Gradient = ', line)[1].split()[0]
                grms, gmax = (float(i) for i in grms_gmax_word.split('/'))
                energy = float(line.split()[-1])
                df_energy.loc[step, label] = energy
                df_grms.loc[step, label] = grms
                df_gmax.loc[step, label] = gmax
                df_drms.loc[step, label] = np.nan
                df_dmax.loc[step, label] = np.nan
                df_de.loc[step, label] = np.nan
            else:
                energy = float(re.split('E \(change\) = ', line)[1].split()[0])
                if "Grad_T" in line: # It's a constrained optimization
                    grms_gmax_word = re.split('Grad_T = ', line)[1].split()[0]
                else:
                    grms_gmax_word = re.split('Grad = ', line)[1].split()[0]
                grms, gmax = (float(i) for i in grms_gmax_word.split('/'))
                drms_dmax_word = re.split('Displace = ', line)[1].split()[0]
                drms, dmax = (float(i) for i in drms_dmax_word.split('/'))
                de = float(re.split('E \(change\) = ', line)[1].split()[1].replace('(','').replace(')',''))
                df_energy.loc[step, label] = energy
                df_grms.loc[step, label] = grms
                df_gmax.loc[step, label] = gmax
                df_drms.loc[step, label] = drms
                df_dmax.loc[step, label] = dmax
                df_de.loc[step, label] = de
            step += 1
        if "Maximum iterations reached" in line:
            if status[label] not in ["Unknown", "MaxIter"]: print("Warning: Found multiple status messages")
            status[label] = "MaxIter"
        if "Converged!" in line:
            if status[label] not in ["Unknown", "Converged"]: print("Warning: Found multiple status messages")
            status[label] = "Converged"
        if "KeyboardInterrupt" in line:
            if status[label] not in ["Unknown", "Interrupted"]: print("Warning: Found multiple status messages")
            status[label] = "Interrupted"

def get_vmin_vmax_log(df, pad=0.2):
    vmin = 10**np.floor(np.log10(np.nanmin(df.replace(0, np.nan).values))-pad)
    vmax = 10**np.ceil(np.log10(np.nanmax(df.replace(0, np.nan).values))+pad)
    return vmin, vmax
    
if np.std(df_energy.iloc[0]) > 1e-6:
    print("--== Warning - step 0 energies are not all the same ==--")

# Convert raw energies to energy change from first frame in kcal/mol
df_energy_kcal = (df_energy - df_energy.iloc[0]).apply(lambda x: x*627.51)
df_de_abs = df_de.apply(lambda x:np.abs(x))

with PdfPages('plot-performance.pdf') as pdf:
    if True:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
        fig.set_size_inches((12, 8))
        fig.subplots_adjust(0.05, 0.15, 0.95, 0.9)
        fig.suptitle(fig_title)
        ax1.set_title('Energy change, from start (kcal/mol)')
        df_energy_kcal.plot(ax=ax1, legend=False)
        labels = []
        final_es = []
        for col in df_energy_kcal.columns:
            labels.append("%s %s N=%i" % (col, status[col], df_energy_kcal[col].last_valid_index()))
            final_es.append(df_energy_kcal[col][df_energy_kcal[col].last_valid_index()])
        final_es = np.array(final_es)
        fig.legend(labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=2)
        ax1in = ax1.inset_axes([0.5, 0.5, 0.5, 0.5])
        ymin = np.min(final_es) - 1
        ymax = np.max(final_es) + 1
        xmin = len(df_energy_kcal)//2
        xmax = len(df_energy_kcal)
        ax1in.set_ylim(ymin, ymax)
        ax1in.set_xlim(xmin, xmax)
        df_energy_kcal.plot(ax=ax1in, legend=False)
        bbox = dict(boxstyle="round", ec="white", fc="white", alpha=0.5)
        plt.setp(ax1in.get_xticklabels(), bbox=bbox)
        plt.setp(ax1in.get_yticklabels(), bbox=bbox)

        ax5.set_xlabel('Optimization Cycle')
        titles = ['RMS Gradient (a.u.)', 'Max Gradient (a.u.)', 'Energy change per step (a.u.)', 'RMS Displacement (a.u.)', 'Max Displacement (a.u.)']
        dfs = [df_grms, df_gmax, df_de_abs, df_drms, df_dmax]
        convs = [3.0e-4, 4.5e-4, 1.0e-6, 1.2e-3, 1.8e-3]
        axs = [ax2, ax3, ax4, ax5, ax6]
        vmins = [get_vmin_vmax_log(df)[0] for df in dfs]
        vmaxs = [get_vmin_vmax_log(df)[1] for df in dfs]
        # RMS and Max should share the same vmin/vmax
        vmins[1] = vmins[0]
        vmaxs[0] = vmaxs[1]
        vmins[4] = vmins[3]
        vmaxs[3] = vmaxs[4]
        for title, df, conv, ax, vmin, vmax in zip(titles, dfs, convs, axs, vmins, vmaxs):
            # fig, ax = plt.subplots()
            # fig.set_size_inches((6, 5))
            # ax.set_xlabel('Optimization Cycle')
            ax.set_title(title)
            # vmin, vmax = get_vmin_vmax_log(df)
            df.plot(ax=ax, ylim=(vmin, vmax), logy=True, legend=False)
            ax.hlines(y=conv, xmin=0, xmax=df.shape[0], colors='k', linestyle='--', linewidth=0.5)

        pdf.savefig(fig)
        plt.close()

    else:
        # Old, multi-page plot style
        # Plot the energy change
        fig, ax = plt.subplots()
        fig.set_size_inches((6, 5))
        ax.set_xlabel('Optimization Cycle')
        ax.set_ylabel('Energy change (kcal/mol)')
        df_energy_kcal.plot(ax=ax)
        labels = []
        for col in df_energy_kcal.columns:
            labels.append("%s %s N=%i" % (col, status[col], df_energy_kcal[col].last_valid_index()))
        ax.legend(labels)
        pdf.savefig(fig)
        plt.close()
    
        titles = ['RMS Gradient', 'Max Gradient', 'RMS Displacement', 'Max Displacement']
        dfs = [df_grms, df_gmax, df_drms, df_dmax]
        convs = [3.0e-4, 4.5e-4, 1.2e-3, 1.8e-3]
        for title, df, conv in zip(titles, dfs, convs):
            fig, ax = plt.subplots()
            fig.set_size_inches((6, 5))
            ax.set_xlabel('Optimization Cycle')
            ax.set_ylabel(title)
            vmin, vmax = get_vmin_vmax_log(df)
            df.plot(ax=ax, ylim=(vmin, vmax), logy=True)
            ax.legend(labels)
            ax.hlines(y=conv, xmin=0, xmax=df.shape[0], colors='k', linestyle='--')
            pdf.savefig(fig)
            plt.close()
