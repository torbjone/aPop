import matplotlib.pyplot as plt
import numpy as np


def return_F(r_star, r, F0):
    F = np.zeros(len(r))
    F[np.where(r < r_star)] = np.sqrt(r[0] / r[np.where(r < r_star)])
    F[np.where(r >= r_star)] = np.sqrt(r[0] / r_star) * (r_star / r[np.where(r >= r_star)])**2
    return F0 * F


def return_r_star(F, r):
    r_stars = np.linspace(10, 1000, 1000)
    errors = np.zeros(len(r_stars))
    for idx, r_star in enumerate(r_stars):
        F_approx = return_F(r_star, r, F[0])
        errors[idx] = np.sum((F - F_approx)**2)
    return r_stars[np.argmin(errors)]

F0 = 10
r_star = 200.2
r_e = 10.
r = 10 * np.exp(np.linspace(0, np.log(1000), 30))
F = return_F(r_star, r, F0)
print r

# for idx in range(len(F)):
#     if r[idx] < r_star:
#         F[idx] = F0 * np.sqrt(r_e / r[idx])
#     else:
#         F[idx] = F0 * np.sqrt(r_e / r_star) * (r_star / r[idx])**2
F *= np.random.normal(1, 0.2, size=len(F))

r_star_approx = return_r_star(F, r)
F_approx = return_F(r_star_approx, r, F0)
print r_star, r_star_approx
plt.loglog(r, F, 'k', lw=2)
plt.loglog(r, F_approx, 'gray', lw=2)
plt.plot([r_star_approx, r_star_approx], [np.max(F), np.min(F)], '--')
plt.show()
