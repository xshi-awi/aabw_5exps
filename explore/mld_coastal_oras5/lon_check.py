"""Longitude of the deepest JJA mixed layers south of 60S (checks the R2/R11 caption claim)."""
import numpy as np
exec(open('coastal_mld.py').read().split("print(f'ORAS5 JJA months")[0])
for name, D in [('MODEL', mod), ('ORAS5', obs)]:
    A = D['area']
    for thr in (200, 300):
        k = D['mld'] > thr
        h, e = np.histogram(D['lon'][k], bins=np.arange(-180, 181, 20), weights=A[k])
        print(f'{name} MLD>{thr} area share by 20-deg longitude bin (%):')
        print('   ' + ' '.join(f'{int(a)}:{100*v/h.sum():.0f}' for a, v in zip(e[:-1], h) if v > 0))
    # max of the 2x2-degree box mean MLD and where
    o = np.argsort(D['mld'])[::-1][:5]
    print('  deepest points:', [(round(float(D['lat'][i]), 1), round(float(D['lon'][i]), 1), int(D['mld'][i])) for i in o])
    for sec, a, b in [('Atlantic 60W-20E', -60, 20), ('140-60W', -140, -60)]:
        s = (D['lon'] >= a) & (D['lon'] < b)
        print(f'  {sec}: mean {np.average(D["mld"][s], weights=A[s]):.0f} m, p99 {np.percentile(D["mld"][s], 99):.0f} m')
