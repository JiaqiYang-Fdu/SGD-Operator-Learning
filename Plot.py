import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from mpl_toolkits.axes_grid1 import make_axes_locatable


# === Load data ===
y_true = np.load("true_field.npy")
y_pred = np.load("predicted_field.npy")
eval_steps = np.load("eval_steps.npy")
test_errors = np.load("test_errors.npy")
test_l2_errors = np.load("test_l2_errors.npy")

# === Log-log linear regression fit for error curves ===
log_x = np.log10(eval_steps)

# Fit for L2 error
log_y1 = np.log10(test_l2_errors)
slope1, intercept1, *_ = linregress(log_x[4:-1], log_y1[4:-1])
fit_line1 = 10 ** (intercept1 + slope1 * log_x)

# Fit for relative error
log_y2 = np.log10(test_errors)
slope2, intercept2, *_ = linregress(log_x[4:-1], log_y2[4:-1])
fit_line2 = 10 ** (intercept2 + slope2 * log_x)

# === Plot 1: Error convergence curves ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.subplots_adjust(wspace=0.4)


# Subplot 1: L2 Error
ax1.loglog(eval_steps, test_l2_errors, label="$L^2$ Error", color='tab:orange',
           marker="o", linewidth=1, markersize=2)
ax1.loglog(eval_steps, fit_line1, label=f"Linear Fit: slope = {slope1:.2f}",
           color='black', linestyle="-.", linewidth=2)
ax1.set_xlabel("Iteration", fontsize=16)
ax1.set_ylabel("$L^2$ Error", fontsize=16)
ax1.set_title("$L^2$ Error vs. Iteration", fontsize=16)
ax1.legend(fontsize=16)
ax1.grid(True, which="both", ls="--", alpha=0.6)

# Subplot 2: Relative Error
ax2.loglog(eval_steps, test_errors, label="Relative Error", color='tab:orange',
           marker="o", linewidth=1, markersize=2)
ax2.loglog(eval_steps, fit_line2, label=f"Linear Fit: slope = {slope2:.2f}",
           color='black', linestyle="-.", linewidth=2)
ax2.set_xlabel("Iteration", fontsize=16)
ax2.set_ylabel("Relative Error", fontsize=16)
ax2.set_title("Relative Error vs. Iteration", fontsize=16)
ax2.legend(fontsize=16)
ax2.grid(True, which="both", ls="--", alpha=0.6)


# plt.tight_layout()
plt.savefig("error_convergence_comparison.png", dpi=300)
plt.show()

# === Plot 2: Field comparison (True, Predicted, Error) ===
extent = [0, 2 * np.pi, 0, 2 * np.pi]
tick_vals = [0.0, 2.5, 5.0]
tick_labels = ["0.0", "2.5", "5.0"]

plt.figure(figsize=(12, 4))

# Subplot 1: True Field
ax1 = plt.subplot(1, 3, 1)
im1 = ax1.imshow(y_true, cmap='viridis', extent=extent, origin='lower')
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_xticks(tick_vals)
ax1.set_xticklabels(tick_labels)
ax1.set_yticks(tick_vals)
ax1.set_yticklabels(tick_labels)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.15)
cb1 = plt.colorbar(im1, cax=cax1)
cb1.set_ticks(np.linspace(-0.075, 0.075, 4))

# Subplot 2: Predicted Field
ax2 = plt.subplot(1, 3, 2)
im2 = ax2.imshow(y_pred, cmap='viridis', extent=extent, origin='lower')
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_xticks(tick_vals)
ax2.set_xticklabels(tick_labels)
ax2.set_yticks(tick_vals)
ax2.set_yticklabels(tick_labels)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.15)
cb2 = plt.colorbar(im2, cax=cax2)
cb2.set_ticks(np.linspace(-0.075, 0.075, 4))

# Subplot 3: Pointwise Error
ax3 = plt.subplot(1, 3, 3)
error_map = y_pred - y_true
im3 = ax3.imshow(error_map, cmap='bwr', extent=extent, origin='lower')
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_xticks(tick_vals)
ax3.set_xticklabels(tick_labels)
ax3.set_yticks(tick_vals)
ax3.set_yticklabels(tick_labels)
divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes("right", size="5%", pad=0.15)
cb3 = plt.colorbar(im3, cax=cax3)
cb3.set_ticks(np.linspace(-0.003, 0.003, 4))

# Add subplot labels below each panel
fig = plt.gcf()
fig.text(0.15, 0.04, "(1) True Field", ha='center', fontsize=14)
fig.text(0.48, 0.04, "(2) Predicted Field", ha='center', fontsize=14)
fig.text(0.81, 0.04, "(3) Pointwise Error", ha='center', fontsize=14)

plt.tight_layout()
plt.savefig("field_comparison.png", dpi=300)
plt.show()