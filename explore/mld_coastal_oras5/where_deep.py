"""Latitude band and sector of the area with deep JJA MLD (supplement to coastal_mld.py)."""
import numpy as np
exec(open('coastal_mld.py').read().split("print(f'ORAS5 JJA months")[0])
SECT = [('Weddell 60W-30E', -60, 30), ('Indian 30-90E', 30, 90), ('W Pacific 90-160E', 90, 160),
        ('Ross 160E-140W', 160, 220), ('Amundsen-Bell. 140-60W', 220, 300)]
for name, D in [('MODEL', mod), ('ORAS5', obs)]:
    A = D['area']; lo = np.mod(D['lon'], 360); near = D['coast'] <= 200
    for thr in (200, 300):
        k = D['mld'] > thr; tot = A[k].sum()
        print(f'\n{name} MLD>{thr}: total {tot/1e12:.3f}e12 m2')
        for b0, b1 in [(-60, -62), (-62, -64), (-64, -66), (-66, -70), (-70, -90)]:
            kk = k & (D['lat'] <= b0) & (D['lat'] > b1)
            print(f'   lat {b0}..{b1}: {100*A[kk].sum()/tot:4.0f}%  (near coast {100*A[kk&near].sum()/tot:3.0f}%)')
        for sn, a, b in SECT:
            s = ((lo >= np.mod(a, 360)) | (lo < b)) if a < 0 else ((lo >= a) & (lo < b))
            print(f'   {sn:24s}: {100*A[k&s].sum()/tot:4.0f}%  (near coast {100*A[k&s&near].sum()/tot:3.0f}%)')
    k = D['mld'] > 200
    print(f'  {name} near-coast area with MLD>200: {A[k&near].sum()/1e12:.3f}e12 m2; >150: {A[(D["mld"]>150)&near].sum()/1e12:.3f}e12 m2')
print()
for name, D in [('MODEL', mod), ('ORAS5', obs)]:
    A = D['area']; near = D['coast'] <= 200; s65 = D['lat'] <= -65
    wed = (D['lon'] >= -60) & (D['lon'] <= 30)
    for thr in (150, 200):
        k = (D['mld'] > thr) & s65
        print(f'{name} south of 65S, MLD>{thr}: {A[k].sum()/1e12:.3f}e12 m2, within 200 km of coast {100*A[k&near].sum()/A[k].sum():.0f}%, offshore Weddell {100*A[k&~near&wed].sum()/A[k].sum():.0f}%')
