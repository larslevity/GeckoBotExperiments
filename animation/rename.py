import os

for jdx, idx in enumerate(range(1, 114)[::-1]):
    os.rename('frame-'+'{}'.format(idx).zfill(3)+'.jpg',
              'frame-'+'{}'.format(114+jdx).zfill(3)+'.jpg')
