import numpy as np
from itertools import product
from functools import partial

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from multiprocessing import Pool

from scipy.interpolate import griddata
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy_healpix import *

from healpy import ring2nest

from virac_utils import *

#from zeropoints_gridded import zeropoint_correcter

akconst = 0.482
intrinsicJK=0.05

def fit_rc_peak(colour, magnitude, 
                colourerr1, colourerr2, 
                intrinsiccolourwidth,
                median_b,thresh=0.2,
                minb=2.):
    data = np.atleast_2d(colour).T
    dataK = np.atleast_2d(magnitude).T
    bic = 1e30
    Nrepeats=3
    for n in range(1, np.min([len(data), 10])):
        bictmp = 1e30
        model = None
        for ii in range(Nrepeats):
            modelT = GaussianMixture(n).fit(data)
            bicC = modelT.bic(data)
            if bicC < bictmp:
                model = modelT
                bictmp = bicC
        bic0 = model.bic(data)
        if (bic0 > bic):
            break
        bic = bic0
    upper_percentile=99.5
    xx = np.linspace(np.percentile(data, 0.5), np.percentile(data, upper_percentile), 5000)
    xmaxloc = np.argmin(np.abs(xx-np.percentile(data,95.)))
    varjk = model.covariances_[:, 0, 0][np.argmax(model.weights_)]
    pp = model.predict(data)
#     subset = v[fltr].reset_index(drop=True)
    varK = np.nanmedian(colourerr2[pp == np.argmax(model.weights_)])**2
    varJ = np.nanmedian(colourerr1[pp == np.argmax(model.weights_)])**2
    sigmajk = varjk - varK - varJ - intrinsiccolourwidth**2
    if sigmajk>0.:
        sigmajk=np.sqrt(sigmajk)
    else:
        sigmajk=0.
    yy = np.exp(model.score_samples(np.atleast_2d(xx).T).T)
    yy = gaussian_filter1d(yy, 0.005/np.diff(xx)[0])
    # return xx,yy
    # Find peak and FWHM from Gaussian fit
    peakloc = np.argmax(yy)
    # Added in this to avoid the case where the turn-off peak is incorrectly identified as the red clump
    # It is probably not completely robust! -- in particular, the choice of using this logic is
    # the peak is less than 20% of the full range.
    # However, we only use this if |b|<2 deg
#     print(xx[peakloc],xx[0],xx[-1],(xx[peakloc] - xx[0]) / (xx[-1] - xx[0]) )
    if ((xx[peakloc] - xx[0]) / (xx[xmaxloc] - xx[0]) < thresh
            and np.abs(median_b) < minb):
        localmin = argrelextrema(yy, np.less)
        tpeakloc = argrelextrema(yy, np.greater)
        if len(localmin) > 0 and len(tpeakloc) > 0:
            localmin = localmin[0]
            tpeakloc = tpeakloc[0]
            if len(localmin) > 0:
                tpeakloc = tpeakloc[tpeakloc > localmin[0]]
                if len(tpeakloc) > 0:
                    peakloc = tpeakloc[-1]
    jk = xx[peakloc]
    if peakloc != np.argmax(yy):
        print('Turn-off peak detected')
#     print(xx[peakloc])
    if(peakloc==0 or peakloc==len(xx)-1):
        return jk, np.nan, xx, yy, data, dataK
    flower = yy[peakloc - 1]
    fupper = yy[peakloc + 1]
    n = -2
    while (flower / yy[peakloc] > .5 and peakloc + n >= 0):
        flower = yy[peakloc + n]
        n -= 1
    xl = xx[peakloc + n + 1]
    n = 2
    while (fupper / yy[peakloc] > .5 and peakloc + n < len(yy)):
        fupper = yy[peakloc + n]
        n += 1
    xu = xx[peakloc + n - 1]
    varjk = ((xu - xl) / 2.355)**2
    sigmajk = varjk - varK - varJ - intrinsiccolourwidth**2
    if sigmajk>0.:
        sigmajk=np.sqrt(sigmajk)
    else:
        sigmajk=0.

    return jk,sigmajk,\
           xx,yy,data,dataK


def fit_rc_peak_JK_virac1(v):
    v = v.reset_index(drop=True)
    fltr = (v.mag < 14. +
            (v.jmag - v.mag - 0.62) * akconst) & (v.jmag - v.mag > 0.5)
    return fit_rc_peak((v.jmag - v.mag).values[fltr], 
                       v.mag.values[fltr], 
                       v.ejmag.values[fltr],
                       v.emag.values[fltr],
                       intrinsiccolourwidth=intrinsicJK,
                       median_b=np.median(v['b'].values[fltr]),
                       thresh=0.2,minb=2.)

def fit_rc_peak_JK_virac2(v):
    v = v.reset_index(drop=True)
    fltr = (v.kmag < 14. +
            (v.jmag - v.kmag - 0.62) * akconst) 
    fltr &= (v.kmag > 11.5 +
            (v.jmag - v.kmag - 0.62) * akconst) 
    fltr &= (v.jmag - v.kmag > 0.5)
    return fit_rc_peak((v.jmag - v.kmag).values[fltr], 
                       v.kmag.values[fltr], 
                       v.ejmag.values[fltr],
                       v.ekmag.values[fltr],
                       intrinsiccolourwidth=intrinsicJK,
                       median_b=np.median(v['b'].values[fltr]),
                       thresh=0.2,minb=2.)

intrinsicHK=0.02
def fit_rc_peak_HK_virac2(v, thresh=0.1):
    v = v.reset_index(drop=True)
    ahconst = 1.13
    fltr = (v.kmag < 14. +
            (v.hmag - v.kmag) * ahconst) 
    fltr &= (v.kmag > 11.5 +
            (v.hmag - v.kmag) * ahconst) 
    fltr &= (v.hmag - v.kmag > 0.)
    return fit_rc_peak((v.hmag - v.kmag).values[fltr], 
                       v.kmag.values[fltr], 
                       v.ehmag.values[fltr],
                       v.ekmag.values[fltr],
                       intrinsiccolourwidth=intrinsicHK,
                       median_b=np.median(v['b'].values[fltr]),
                       thresh=thresh,minb=1.5)


#extra_cat = pd.concat([
#		pd.read_hdf('/local/scratch_1/jls/virac/b388_J_catalogue.hdf5','data'),
#		pd.read_hdf('/local/scratch_1/jls/virac/b212_J_catalogue.hdf5','data')
#		])

JK_ref_colour = 0.62
JH_ref_colour = 0.54
HK_ref_colour = 0.09

# data_fn = {'v1_JK':grab_virac_wsdb_extinction_cut,
#            'v2_JK':grab_virac_wsdb_v2_JK_extinction_cut,
#            'v2_HK':grab_virac_wsdb_v2_HK_extinction_cut,
#            'v2_NEW_JK':grab_virac_wsdb_v2_NEW_JK_extinction_cut,
#            'v2_NEW_HK':grab_virac_wsdb_v2_NEW_HK_extinction_cut}

# tmass_data_fn = {'v1_JK':grab_2mass_wsdb_extinction_cut,
#            'v2_JK':grab_2mass_wsdb_JK_extinction_cut,
#            'v2_HK':grab_2mass_wsdb_HK_extinction_cut}

# ref_colour = {'v1_JK':JK_ref_colour,
#            'v2_JK':JK_ref_colour,
#            'v2_HK':HK_ref_colour,
#            'v2_NEW_JK':JK_ref_colour,
#            'v2_NEW_HK':HK_ref_colour,
#            'v2_NEW_JK_UBERCAL':JK_ref_colour,
#            'v2_NEW_HK_UBERCAL':HK_ref_colour,
#            'v2_NEW_JK_PUBLIC':JK_ref_colour,
#            'v2_NEW_HK_PUBLIC':HK_ref_colour}

# fitting_fn = {'v1_JK':fit_rc_peak_JK_virac1,
#            'v2_JK':fit_rc_peak_JK_virac2,
#            'v2_HK':fit_rc_peak_HK_virac2,
#            'v2_NEW_JK':fit_rc_peak_JK_virac2,
#            'v2_NEW_HK':fit_rc_peak_HK_virac2,
#            'v2_NEW_JK_UBERCAL':fit_rc_peak_JK_virac2,
#            'v2_NEW_HK_UBERCAL':fit_rc_peak_HK_virac2,
#            'v2_NEW_JK_PUBLIC':fit_rc_peak_JK_virac2,
#            'v2_NEW_HK_PUBLIC':fit_rc_peak_HK_virac2}

def find_extinction(version, size, arr):
    all_data = data_fn[version](arr[0], arr[1], size, 14.)
    if (len(all_data) > 0):
        result = fitting_fn[version](all_data)
        ext = result[0] - ref_colour[version]
        ext_spread = result[1]
    else:
        # Should only enter in v1 when extra catalogues were needed
        all_data = grab_virac_wsdb_all(arr[0], arr[1], size)
        c = SkyCoord(ra=all_data.ra.values * u.degree,
                     dec=all_data.de.values * u.degree)
        catalog = SkyCoord(ra=extra_cat.RA.values * u.degree,
                           dec=extra_cat.Dec.values * u.degree)
        idx, d2d, d3d = c.match_to_catalog_sky(catalog)
        jj = extra_cat.mag.values[idx]
        jj[d2d.arcsec > 0.8] = 99.9
        ejj = extra_cat.emag.values[idx]
        ejj[d2d.arcsec > 0.8] = 99.9
        all_data['jmag'] = jj
        all_data['ejmag'] = ejj
        fltr = (all_data.mag.values > 11.5) & (all_data.mag.values < 20.) & (
            all_data.jmag.values < 20.)
        fltr &= (all_data.emag.values <
                 0.2) & (all_data.ejmag.values < 0.2) & (
                     all_data.jmag.values - all_data.mag.values > 0.5)
        fltr &= (all_data.mag.values < 14. +
                 (all_data.jmag.values - all_data.mag.values - 0.62) * 0.482)
        all_data = all_data[fltr].reset_index(drop=True)
        if (len(all_data) == 0):
            ext = np.nan
            ext_spread = np.nan
        else:
            result = fit_rc_peak_JK_virac1(all_data)
            ext = result[0] - 0.62
            ext_spread = result[1]

    if ext < 0.:
        ext = 0.
    #print arr[0],arr[1],ext, ext_spread
    return [arr[0], arr[1], ext, ext_spread]


def find_tmass_extinction(version, size, arr):
    all_data = tmass_data_fn[version](arr[0], arr[1], size, 14.)
    if (len(all_data) > 0):
        result = fitting_fn[version](all_data)
        ext = result[0] - ref_colour[version]
        ext_spread = result[1]
        if ext < 0.:
            ext = 0.
#         print(arr[0], arr[1], ext, ext_spread)
        return [arr[0], arr[1], ext, ext_spread]
    else:
        return [arr[0], arr[1], 0., np.nan]

def generate_healpix_NEW():
    query = """
    select avg(l), avg(b), count(*), healpix_ang2ipix_ring(256,l,b)
    from leigh_smith.virac2_public 
    where (l>350. or l<10) 
    group by healpix_ang2ipix_ring(256,l,b);"""

    fullV = pd.DataFrame(sqlutil.get(query,host='cappc127',user='jason_sanders',
                                     password=wsdbpassword,asDict=True))
    fullV.columns = ['l','b','count','hlpx']
    fullV['hlpx'].to_csv("extmap_healpix_NEW.csv")
    
def generate_healpix():
    query = """
    select avg(l), avg(b), count(*), healpix_ang2ipix_ring(256,l,b)
    from leigh_smith.virac_pm2 
    where (l>350. or l<10) and ekmag<0.2 and ejmag<0.2 
    and jmag-kmag>0.5
    and kmag>11.5+0.482*(jmag-kmag-0.62)
    and kmag<14.+0.482*(jmag-kmag-0.62)
    group by healpix_ang2ipix_ring(256,l,b);"""

    fullV = pd.DataFrame(sqlutil.get(query,host='cappc127',user='jason_sanders',
                                     password=wsdbpassword,asDict=True))
    fullV.columns = ['l','b','count','hlpx']
    fullV['hlpx'].to_csv("extmap_healpix.csv")
    
def grab_indiv_JK(h):
    l,b = healpix_to_lonlat(h, nside=256)
    l,b = l.to(u.deg).value, b.to(u.deg).value
    
    queryV = """
        select l,b,jmag,kmag,ejmag,ekmag, pmra, pmdec
        from leigh_smith.virac_pm2 
        where 
        %s
        and kmag>11.5 and kmag<20 
        and jmag>0. and jmag<20
        and ekmag<0.2 and ejmag<0.2 
        and jmag-kmag>0.5
        and kmag>11.5+0.482*(jmag-kmag-0.62)
        and kmag<14.+0.482*(jmag-kmag-0.62)
        and duplicate=0
        ;"""

    size = 30.
    size /= 60.
    poly_string = "l>%0.3f and l<%0.3f and b>%0.3f and b<%0.3f"\
                    %(l-.5*size,l+.5*size,b-.5*size,b+.5*size)

    if (l - .5 * size < 0.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size+360.,l+.5*size,b-.5*size,b+.5*size)
    if (l + .5 * size > 360.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size,l+.5*size-360.,b-.5*size,b+.5*size)
        
    df=pd.DataFrame(sqlutil.get(queryV%poly_string,**wsdb_kwargs))
    hpx = lonlat_to_healpix(df['l'].values*u.deg,df['b'].values*u.deg, 256)
    print(l,b,h,len(df))
    return df[hpx==h].reset_index(drop=True)    

def grab_indiv_HK(h):
    l,b = healpix_to_lonlat(h, nside=256)
    l,b = l.to(u.deg).value, b.to(u.deg).value
    
    queryV = """
        select l,b,hmag,kmag,ehmag,ekmag,pmra,pmdec
        from leigh_smith.virac_pm2 
        where 
        %s
        and kmag>11.5 and kmag<20 
        and hmag>0. and hmag<20
        and ekmag<0.2 and ehmag<0.2 
        and hmag-kmag>0.
        and kmag>11.5+1.13*(hmag-kmag)
        and kmag<14.+1.13*(hmag-kmag)
        and duplicate=0
        ;"""

    size = 30.
    size /= 60.
    poly_string = "l>%0.3f and l<%0.3f and b>%0.3f and b<%0.3f"\
                    %(l-.5*size,l+.5*size,b-.5*size,b+.5*size)

    if (l - .5 * size < 0.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size+360.,l+.5*size,b-.5*size,b+.5*size)
    if (l + .5 * size > 360.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size,l+.5*size-360.,b-.5*size,b+.5*size)
        
    df=pd.DataFrame(sqlutil.get(queryV%poly_string,**wsdb_kwargs))
    hpx = lonlat_to_healpix(df['l'].values*u.deg,df['b'].values*u.deg, 256)
    print(l,b,h,len(df))
    return df[hpx==h].reset_index(drop=True)    

def grab_indiv_JK_NEW(h):
    
    l,b = healpix_to_lonlat(h, nside=256)
    l,b = l.to(u.deg).value, b.to(u.deg).value
#    s = SkyCoord(l=l,b=b,frame='galactic')
#    ra, dec = s.transform_to('icrs').ra.deg, s.transform_to('icrs').dec.deg
    
    queryV = """
        select l,b,
        j_ivw_mean_mag as jmag,
        ks_ivw_mean_mag as kmag,
        j_ivw_err_mag as ejmag,
        ks_ivw_err_mag as ekmag,
        pmra, pmdec
        from leigh_smith.virac2
        where 
        %s
        and ks_ivw_mean_mag>11.5 and ks_ivw_mean_mag<20 
        and j_ivw_mean_mag>0. and j_ivw_mean_mag<20
        and ks_ivw_err_mag<0.2 and j_ivw_err_mag<0.2 
        and j_ivw_mean_mag-ks_ivw_mean_mag>0.5
        and ks_ivw_mean_mag>11.5+0.482*(j_ivw_mean_mag-ks_ivw_mean_mag-0.62)
        and ks_ivw_mean_mag<14.+0.482*(j_ivw_mean_mag-ks_ivw_mean_mag-0.62)
        ;"""

    size = 30.
    size /= 60.
    poly_string = "l>%0.3f and l<%0.3f and b>%0.3f and b<%0.3f"\
                    %(l-.5*size,l+.5*size,b-.5*size,b+.5*size)
    if (l - .5 * size < 0.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size+360.,l+.5*size,b-.5*size,b+.5*size)
    if (l + .5 * size > 360.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size,l+.5*size-360.,b-.5*size,b+.5*size)
        
        
#    poly_string = "ra>%0.3f and ra<%0.3f and dec>%0.3f and dec<%0.3f"\
#                    %(ra-.5*size,ra+.5*size,dec-.5*size,dec+.5*size)
        
    df=pd.DataFrame(sqlutil.get(queryV%poly_string,**wsdb_kwargs))
    if(len(df)==0):
        return df
#    c = SkyCoord(ra=df['ra'].values*u.deg,dec=df['dec'].values*u.deg)
#    df['l'], df['b'] = c.transform_to('galactic').l.deg, c.transform_to('galactic').b.deg
    
    hpx = lonlat_to_healpix(df['l'].values*u.deg,df['b'].values*u.deg, 256)
#    l,b = l.to(u.deg).value, b.to(u.deg).value
    print(l,b,h,len(df))
    return df[hpx==h].reset_index(drop=True)    

def grab_indiv_HK_NEW(h):
    l,b = healpix_to_lonlat(h, nside=256)
    l,b = l.to(u.deg).value, b.to(u.deg).value
    #s = SkyCoord(l=l,b=b,frame='galactic')
    #ra, dec = s.transform_to('icrs').ra.deg, s.transform_to('icrs').dec.deg   
    
    queryV = """
        select l,b,
        h_ivw_mean_mag as hmag,
        ks_ivw_mean_mag as kmag,
        h_ivw_err_mag as ehmag,
        ks_ivw_err_mag as ekmag,
        pmra,pmdec
        from leigh_smith.virac2
        where 
        %s
        and ks_ivw_mean_mag>11.5 and ks_ivw_mean_mag<20 
        and h_ivw_mean_mag>0. and h_ivw_mean_mag<20
        and ks_ivw_err_mag<0.2 and h_ivw_err_mag<0.2 
        and h_ivw_mean_mag-ks_ivw_mean_mag>0.
        and ks_ivw_mean_mag>11.5+1.13*(h_ivw_mean_mag-ks_ivw_mean_mag)
        and ks_ivw_mean_mag<14.+1.13*(h_ivw_mean_mag-ks_ivw_mean_mag)
        ;"""

    size = 30.
    size /= 60.
    poly_string = "l>%0.3f and l<%0.3f and b>%0.3f and b<%0.3f"\
                    %(l-.5*size,l+.5*size,b-.5*size,b+.5*size)
    if (l - .5 * size < 0.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size+360.,l+.5*size,b-.5*size,b+.5*size)
    if (l + .5 * size > 360.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size,l+.5*size-360.,b-.5*size,b+.5*size)
        
        
#    poly_string = "ra>%0.3f and ra<%0.3f and dec>%0.3f and dec<%0.3f"\
#                    %(ra-.5*size,ra+.5*size,dec-.5*size,dec+.5*size)
        
    df=pd.DataFrame(sqlutil.get(queryV%poly_string,**wsdb_kwargs))
    if(len(df)==0):
        return df
#    c = SkyCoord(ra=df['ra'].values*u.deg,dec=df['dec'].values*u.deg)
#    df['l'], df['b'] = c.transform_to('galactic').l.deg, c.transform_to('galactic').b.deg
    
    hpx = lonlat_to_healpix(df['l'].values*u.deg,df['b'].values*u.deg, 256)
#    l,b = l.to(u.deg).value, b.to(u.deg).value
    print(l,b,h,len(df))
    return df[hpx==h].reset_index(drop=True)  


def grab_indiv_JK_NEW_UBERCAL(h):
    
    l,b = healpix_to_lonlat(h, nside=256)
    l,b = l.to(u.deg).value, b.to(u.deg).value
    
    queryV = """
        select l,b,
        s.j_g_ivw_mean_mag as jmag,
        s.ks_g_ivw_mean_mag as kmag,
        s.j_g_ivw_err_mag as ejmag,
        s.ks_g_ivw_err_mag as ekmag,
        v.pmra, v.pmdec
        from leigh_smith.virac2 as v inner join leigh_smith.virac2_uberstats as s on v.sourceid=s.sourceid
        where 
        %s
        and s.ks_g_ivw_mean_mag>11.5 and s.ks_g_ivw_mean_mag<20 
        and s.j_g_ivw_mean_mag>0. and s.j_g_ivw_mean_mag<20
        and s.ks_g_ivw_err_mag<0.2 and s.j_g_ivw_err_mag<0.2 
        and s.j_g_ivw_mean_mag-s.ks_g_ivw_mean_mag>0.5
        and s.ks_g_ivw_mean_mag>11.5+0.482*(s.j_g_ivw_mean_mag-s.ks_g_ivw_mean_mag-0.62)
        and s.ks_g_ivw_mean_mag<14.+0.482*(s.j_g_ivw_mean_mag-s.ks_g_ivw_mean_mag-0.62)
        ;"""

    size = 30.
    size /= 60.
    poly_string = "l>%0.3f and l<%0.3f and b>%0.3f and b<%0.3f"\
                    %(l-.5*size,l+.5*size,b-.5*size,b+.5*size)
    if (l - .5 * size < 0.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size+360.,l+.5*size,b-.5*size,b+.5*size)
    if (l + .5 * size > 360.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size,l+.5*size-360.,b-.5*size,b+.5*size)

    df=pd.DataFrame(sqlutil.get(queryV%poly_string,**wsdb_kwargs))
    if(len(df)==0):
        return df
    
    hpx = lonlat_to_healpix(df['l'].values*u.deg,df['b'].values*u.deg, 256)
    print(l,b,h,len(df))
    return df[hpx==h].reset_index(drop=True)    

def grab_indiv_HK_NEW_UBERCAL(h):
    l,b = healpix_to_lonlat(h, nside=256)
    l,b = l.to(u.deg).value, b.to(u.deg).value
    
    queryV = """
        select l,b,
        s.h_g_ivw_mean_mag as hmag,
        s.ks_g_ivw_mean_mag as kmag,
        s.h_g_ivw_err_mag as ehmag,
        s.ks_g_ivw_err_mag as ekmag,
        v.pmra, v.pmdec
        from leigh_smith.virac2 as v inner join leigh_smith.virac2_uberstats as s on v.sourceid=s.sourceid
        where 
        %s
        and s.ks_g_ivw_mean_mag>11.5 and s.ks_g_ivw_mean_mag<20 
        and s.h_g_ivw_mean_mag>0. and s.h_g_ivw_mean_mag<20
        and s.ks_g_ivw_err_mag<0.2 and s.h_g_ivw_err_mag<0.2 
        and s.h_g_ivw_mean_mag-s.ks_g_ivw_mean_mag>0.
        and s.ks_g_ivw_mean_mag>11.5+1.13*(s.h_g_ivw_mean_mag-s.ks_g_ivw_mean_mag)
        and s.ks_g_ivw_mean_mag<14.+1.13*(s.h_g_ivw_mean_mag-s.ks_g_ivw_mean_mag)
        ;"""

    size = 30.
    size /= 60.
    poly_string = "l>%0.3f and l<%0.3f and b>%0.3f and b<%0.3f"\
                    %(l-.5*size,l+.5*size,b-.5*size,b+.5*size)
    if (l - .5 * size < 0.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size+360.,l+.5*size,b-.5*size,b+.5*size)
    if (l + .5 * size > 360.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size,l+.5*size-360.,b-.5*size,b+.5*size)

    df=pd.DataFrame(sqlutil.get(queryV%poly_string,**wsdb_kwargs))
    if(len(df)==0):
        return df
    
    hpx = lonlat_to_healpix(df['l'].values*u.deg,df['b'].values*u.deg, 256)
    print(l,b,h,len(df))
    return df[hpx==h].reset_index(drop=True)  

def grab_indiv_JK_NEW_PUBLIC(h):
    
    l,b = healpix_to_lonlat(h, nside=256)
    l,b = l.to(u.deg).value, b.to(u.deg).value
    
    queryV = """
        select l,b,
        j_b_ivw_mean_mag as jmag,
        ks_b_ivw_mean_mag as kmag,
        j_b_ivw_err_mag as ejmag,
        ks_b_ivw_err_mag as ekmag,
        pmra, pmdec
        from leigh_smith.virac2_public
        where 
        %s
        and ks_b_ivw_mean_mag>11.5 and ks_b_ivw_mean_mag<20 
        and j_b_ivw_mean_mag>0. and j_b_ivw_mean_mag<20
        and ks_b_ivw_err_mag<0.2 and j_b_ivw_err_mag<0.2 
        and j_b_ivw_mean_mag-ks_b_ivw_mean_mag>0.5
        and ks_b_ivw_mean_mag>11.5+0.482*(j_b_ivw_mean_mag-ks_b_ivw_mean_mag-0.62)
        and ks_b_ivw_mean_mag<14.+0.482*(j_b_ivw_mean_mag-ks_b_ivw_mean_mag-0.62)
        ;"""

    size = 30.
    size /= 60.
    poly_string = "l>%0.3f and l<%0.3f and b>%0.3f and b<%0.3f"\
                    %(l-.5*size,l+.5*size,b-.5*size,b+.5*size)
    if (l - .5 * size < 0.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size+360.,l+.5*size,b-.5*size,b+.5*size)
    if (l + .5 * size > 360.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size,l+.5*size-360.,b-.5*size,b+.5*size)

    df=pd.DataFrame(sqlutil.get(queryV%poly_string,**wsdb_kwargs))
    if(len(df)==0):
        return df
    
    hpx = lonlat_to_healpix(df['l'].values*u.deg,df['b'].values*u.deg, 256)
    print(l,b,h,len(df))
    return df[hpx==h].reset_index(drop=True)  

def grab_indiv_HK_NEW_PUBLIC(h):
    l,b = healpix_to_lonlat(h, nside=256)
    l,b = l.to(u.deg).value, b.to(u.deg).value
    
    queryV = """
        select l,b,
        h_b_ivw_mean_mag as hmag,
        ks_b_ivw_mean_mag as kmag,
        h_b_ivw_err_mag as ehmag,
        ks_b_ivw_err_mag as ekmag,
        pmra, pmdec
        from leigh_smith.virac2_public
        where 
        %s
        and ks_b_ivw_mean_mag>11.5 and ks_b_ivw_mean_mag<20 
        and h_b_ivw_mean_mag>0. and h_b_ivw_mean_mag<20
        and ks_b_ivw_err_mag<0.2 and h_b_ivw_err_mag<0.2 
        and h_b_ivw_mean_mag-ks_b_ivw_mean_mag>0.
        and ks_b_ivw_mean_mag>11.5+1.13*(h_b_ivw_mean_mag-ks_b_ivw_mean_mag)
        and ks_b_ivw_mean_mag<14.+1.13*(h_b_ivw_mean_mag-ks_b_ivw_mean_mag)
        ;"""

    size = 30.
    size /= 60.
    poly_string = "l>%0.3f and l<%0.3f and b>%0.3f and b<%0.3f"\
                    %(l-.5*size,l+.5*size,b-.5*size,b+.5*size)
    if (l - .5 * size < 0.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size+360.,l+.5*size,b-.5*size,b+.5*size)
    if (l + .5 * size > 360.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size,l+.5*size-360.,b-.5*size,b+.5*size)

    df=pd.DataFrame(sqlutil.get(queryV%poly_string,**wsdb_kwargs))
    if(len(df)==0):
        return df
    
    hpx = lonlat_to_healpix(df['l'].values*u.deg,df['b'].values*u.deg, 256)
    print(l,b,h,len(df))
    return df[hpx==h].reset_index(drop=True)  

grab_indiv = {'v2_JK':grab_indiv_JK, 'v2_HK':grab_indiv_HK,
              'v2_NEW_JK':grab_indiv_JK_NEW, 'v2_NEW_HK':grab_indiv_HK_NEW,
              'v2_NEW_JK_UBERCAL':grab_indiv_JK_NEW_UBERCAL, 'v2_NEW_HK_UBERCAL':grab_indiv_HK_NEW_UBERCAL,
              'v2_NEW_JK_PUBLIC':grab_indiv_JK_NEW_PUBLIC, 'v2_NEW_HK_PUBLIC':grab_indiv_HK_NEW_PUBLIC}

def compute_extinction_healpix(version, hpx):
#     lbh = l, b and healpix at nside=256
    # We subdivide provided there are 100 stars in each pixel
    v = grab_indiv[version](int(hpx))
    if(len(v)<=1):
        return np.nan*np.ones((9,1))
    nside = 256
    thresh=100
    uniq = np.zeros_like(v['l'].values, dtype=np.int64)
    while np.any(uniq==0):
        indx=lonlat_to_healpix(v['l'].values*u.deg,
                               v['b'].values*u.deg,
                               nside)
        groups = v[uniq==0].groupby(indx[uniq==0])
        for ii, s in zip(groups.size().index,groups.size()):
            if np.any(v[(indx==ii)&(uniq==0)].groupby(lonlat_to_healpix(v[(indx==ii)&(uniq==0)]['l'].values*u.deg,
                                                                       v[(indx==ii)&(uniq==0)]['b'].values*u.deg,
                                                                        nside*2)).size().values<thresh):    
                uniq[(indx==ii)&(uniq==0)]=level_ipix_to_uniq(nside_to_level(nside), indx)[(indx==ii)&(uniq==0)]
        nside*=2
        if(nside>20000):
            print('FAIL')
            break
    groups = v.groupby(uniq)
    counts = groups.size().values
    rcc = np.array([fitting_fn[version](v[uniq==u]) for u in np.unique(uniq)])
    ee = rcc[:,0]-ref_colour[version]
    l,b=healpix_to_lonlat(uniq_to_level_ipix(groups.size().index)[1],
                             level_to_nside(uniq_to_level_ipix(groups.size().index)[0]))
    resolution=nside_to_pixel_resolution(level_to_nside(uniq_to_level_ipix(groups.size().index)[0]))
    level, ipix = uniq_to_level_ipix(groups.size().index)
    return np.array([l.to(u.deg).value,b.to(u.deg).value,ee,rcc[:,1],counts,resolution.to(u.arcmin).value,level,ipix, np.unique(uniq)])

def build_extinction_map_healpix(version, hpx_file):
    hlpx = pd.read_csv(hpx_file, names=['hlpx'])
    arr = hlpx['hlpx'].values
    p = Pool(100)
    r = p.map(partial(compute_extinction_healpix, version), arr)
    p.close()
    p.join()  
    r = [np.concatenate([r[i][j] for i in range(len(r))]) for j in range(9)]
    results = pd.DataFrame(np.array(r).T, columns=['l', 'b', 
                                       'e%s'%version[-2:].lower(), 
                                       'sigma_e%s'%version[-2:].lower(),
                                       'counts','resolution',
                                       'level','ipix','uniq'])
    #results.to_csv('/data/jls/virac/virac_extinction_map_healpix_%s.csv'%version)
    
def build_extinction_map(version):
    def fill(lchunks, bchunks, size):
        lchunks = .5 * (lchunks[1:] + lchunks[:-1])
        bchunks = .5 * (bchunks[1:] + bchunks[:-1])
        lchunks = 360. * (lchunks < 0.) + lchunks
        arr = np.array(list(product(lchunks, bchunks)))
        p = Pool(150)
        r = p.map(partial(find_extinction, version, size), arr)
        p.close()
        p.join()
        return r

    lchunks = np.linspace(-10., 10., 121)
    bchunks = np.linspace(-10., -5., 31)
    n = fill(lchunks, bchunks, 10.)

    lchunks = np.linspace(-10., 10., 201)
    bchunks = np.linspace(-5., -1.5, 36)
    n2 = fill(lchunks, bchunks, 6.)
    n = np.vstack((n, n2))

    lchunks = np.linspace(-10., 10., 401)
    bchunks = np.linspace(-1.5, 1.5, 61)
    n2 = fill(lchunks, bchunks, 3.)
    n = np.vstack((n, n2))

    lchunks = np.linspace(-10., 10., 201)
    bchunks = np.linspace(1.5, 5., 36)
    n2 = fill(lchunks, bchunks, 6.)
    n = np.vstack((n, n2))

    results = pd.DataFrame(n, columns=['l', 'b', 
                                       'e%s'%version[-2:].lower(), 
                                       'sigma_e%s'%version[-2:].lower()])
    results.to_csv('/data/jls/virac/virac_extinction_map_%s.csv'%version)


def build_disc_extinction_map(version):
    def fill(lchunks, bchunks, size):
        lchunks = .5 * (lchunks[1:] + lchunks[:-1])
        bchunks = .5 * (bchunks[1:] + bchunks[:-1])
        lchunks = 360. * (lchunks < 0.) + lchunks
        arr = np.array(list(product(lchunks, bchunks)))
        p = Pool(150)
        r = p.map(partial(find_extinction, version, size), arr)
        p.close()
        p.join()
        return r

    lchunks = np.linspace(-15., -10., 51)
    bchunks = np.linspace(-2., -1.5, 6)
    n = fill(lchunks, bchunks, 6.)

    lchunks = np.linspace(-15., -10., 101)
    bchunks = np.linspace(-1.5, 1.5, 61)
    n2 = fill(lchunks, bchunks, 3.)
    n = np.vstack((n, n2))

    lchunks = np.linspace(-15., -10., 51)
    bchunks = np.linspace(1.5, 2., 6)
    n2 = fill(lchunks, bchunks, 6.)
    n = np.vstack((n, n2))

    results = pd.DataFrame(n, columns=['l', 'b', 
                                       'e%s'%version[-2:].lower(), 
                                       'sigma_e%s'%version[-2:].lower()])
    results.to_csv('/data/jls/virac/virac_disc_extinction_map_%s.csv'%version)


def build_2mass_extinction_map(version):
    def fill(lchunks, bchunks, size):
        lchunks = .5 * (lchunks[1:] + lchunks[:-1])
        bchunks = .5 * (bchunks[1:] + bchunks[:-1])
        lchunks = 360. * (lchunks < 0.) + lchunks
        arr = np.array(list(product(lchunks, bchunks)))
        p = Pool(150)
        r = p.map(partial(find_tmass_extinction, version, size), arr)
        p.close()
        p.join()
        return r

    lchunks = np.linspace(-10., 10., 121)
    bchunks = np.linspace(5., 10., 31)
    n = fill(lchunks, bchunks, 10.)

    lchunks = np.linspace(-15., -10., 31)
    bchunks = np.linspace(-10., -5., 31)
    n2 = fill(lchunks, bchunks, 10.)
    n = np.vstack((n, n2))

    lchunks = np.linspace(-15., -10., 51)
    bchunks = np.linspace(-5., -2., 31)
    n2 = fill(lchunks, bchunks, 6.)
    n = np.vstack((n, n2))

    #lchunks = np.linspace(-15.,-10.,101)
    #bchunks = np.linspace(-1.5,1.5,61)
    #n2 = fill(lchunks, bchunks, 3.)
    #n = np.vstack((n,n2))

    lchunks = np.linspace(-15., -10., 51)
    bchunks = np.linspace(2., 5., 31)
    n2 = fill(lchunks, bchunks, 6.)
    n = np.vstack((n, n2))

    lchunks = np.linspace(-15., -10., 31)
    bchunks = np.linspace(5., 10., 31)
    n2 = fill(lchunks, bchunks, 10.)
    n = np.vstack((n, n2))

    lchunks = np.linspace(10., 15., 31)
    bchunks = np.linspace(-10., -5., 31)
    n2 = fill(lchunks, bchunks, 10.)
    n = np.vstack((n, n2))

    lchunks = np.linspace(10., 15., 51)
    bchunks = np.linspace(-5., -1.5, 36)
    n2 = fill(lchunks, bchunks, 6.)
    n = np.vstack((n, n2))

    lchunks = np.linspace(10., 15., 101)
    bchunks = np.linspace(-1.5, 1.5, 61)
    n2 = fill(lchunks, bchunks, 3.)
    n = np.vstack((n, n2))

    lchunks = np.linspace(10., 15., 51)
    bchunks = np.linspace(1.5, 5., 36)
    n2 = fill(lchunks, bchunks, 6.)
    n = np.vstack((n, n2))

    lchunks = np.linspace(10., 15., 31)
    bchunks = np.linspace(5., 10., 31)
    n2 = fill(lchunks, bchunks, 10.)
    n = np.vstack((n, n2))

    results = pd.DataFrame(n, columns=['l', 'b', 
		'e%s'%version[-2:].lower(), 'sigma_e%s'%version[-2:].lower()])
    results.to_csv('/data/jls/virac/tmass_extinction_map_%s.csv'%version)


class extinction_map(object):
    def __init__(self, version='JK', add_tmass=False, add_disc=True):
        self.emap = pd.read_csv('/data/jls/virac/virac_extinction_map_v2_%s.csv'%version)
        self.col = version.lower()
        self.emap.loc[self.emap['e%s'%self.col] != self.emap['e%s'%self.col], 'e%s'%self.col] = 0.
        self.emap.loc[self.emap['sigma_e%s'%self.col] != self.
                      emap['sigma_e%s'%self.col], 'sigma_e%s'%self.col] = 0.
        if add_disc:
            vd = pd.read_csv('/data/jls/virac/virac_disc_extinction_map_v2_%s.csv'%version)
            vd.loc[vd['e%s'%self.col] != vd['e%s'%self.col], 'e%s'%self.col] = 0.
            vd.loc[vd['sigma_e%s'%self.col] != vd['sigma_e%s'%self.col], 'sigma_e%s'%self.col] = 0.
            self.emap = pd.concat([self.emap, vd], axis=0)
        if (add_tmass):
            tm = pd.read_csv('/data/jls/virac/tmass_extinction_map_v2_%s.csv'%version)
            tm.loc[tm['e%s'%self.col] != tm['e%s'%self.col], 'e%s'%self.col] = 0.
            tm.loc[tm['sigma_e%s'%self.col] != tm['sigma_e%s'%self.col], 'sigma_e%s'%self.col] = 0.
            self.emap = pd.concat([self.emap, tm], axis=0)
            self.tmass = True
        else:
            self.tmass = False

    def query(self, l, b):
        ''' l,b in deg, returns extinction (either ejk or ehk) '''
        l = l - 360. * (l > 180.)
        return griddata(np.vstack(
            (self.emap['l'].values - 360. * (self.emap['l'].values > 180.),
             self.emap['b'].values)).T,
                        self.emap['e%s'%self.col].values, (l, b),
                        method='nearest')

    def query_spread(self, l, b):
        ''' l,b in deg , returns ejk spread'''
        l = l - 360. * (l > 180.)
        return griddata(np.vstack(
            (self.emap['l'].values - 360. * (self.emap['l'].values > 180.),
             self.emap['b'].values)).T,
                        self.emap['sigma_e%s'%self.col].values, (l, b),
                        method='nearest')

    def extinction_overlay(self, ak=0.8, aecoeff=0.482):
        print(
            'Extinction overlay at AK=0.8 as shows the two dust lanes out of the plane'
        )
        ll,bb=np.linspace(-10.-5.*self.tmass,10.+5.*self.tmass,100+50*self.tmass),\
  np.linspace(-10.,5.+5.*self.tmass,100+30*self.tmass)
        ll, bb = np.meshgrid(ll, bb)
        plt.contourf(ll, bb, self.query(ll, bb), [ak / aecoeff, 10.], alpha=0.5)
        plt.contour(ll,
                    bb,
                    self.query(ll, bb), [ak / aecoeff, 10.],
                    alpha=0.5,
                    colors='grey')

class moc(object):
    def __init__(self, file=None, data=None):
        if data is None and file==None:
            print('Must provide either data or file')
        if file is not None:
            data = pd.read_csv(file)
        fld, uniq, level = data['fld'], data['uniq'], data['level']
        self.fld=fld
        maxlevel = np.max(level)
        self.maxnside = level_to_nside(maxlevel)
        level, ipix = uniq_to_level_ipix(uniq)
        self.index_nest = ring2nest(level_to_nside(level), ipix)
        self.index_nest = self.index_nest * (2**(maxlevel - level))**2
        self.sorter = np.argsort(self.index_nest)
    def query(self, l, b):
        match_ipix = lonlat_to_healpix(l*u.deg, b*u.deg, self.maxnside, order='nested')
        i = self.sorter[np.searchsorted(self.index_nest, match_ipix, side='right', sorter=self.sorter) - 1]
        return self.fld[i].values
        
class extinction_map_healpix(object):
    def __init__(self, version='v2_NEW_JK_PUBLIC', zeropoint_correct=False):
        self.version = version
        self.ff = pd.read_csv('/data/jls/virac/virac_extinction_map_healpix_%s.csv'%self.version)
        self.ff = self.ff[self.ff['level']>0].reset_index(drop=True)
        maxlevel = np.max(self.ff['level'])
        self.maxnside = level_to_nside(maxlevel)
        level, ipix = uniq_to_level_ipix(self.ff['uniq'])
        self.index_nest = ring2nest(level_to_nside(level), ipix)
        self.index_nest = self.index_nest * (2**(maxlevel - level))**2
        self.sorter = np.argsort(self.index_nest)
        if zeropoint_correct:
            zp = zeropoint_correcter('/data/jls/virac/zeropoints_v2.csv')
            self.ff['e%s'%self.version[-2:].lower()]-=(zp.query(self.ff['l'].values,self.ff['b'].values,self.version[-2])
                                                       - zp.query(self.ff['l'].values,self.ff['b'].values,'Ks'))
    def query(self, l, b):
        match_ipix = lonlat_to_healpix(l*u.deg, b*u.deg, self.maxnside, order='nested')
        i = self.sorter[np.searchsorted(self.index_nest, match_ipix, side='right', sorter=self.sorter) - 1]
        return self.ff['e%s'%self.version[-2:].lower()].values[i]
    def query_spread(self, l, b):
        match_ipix = lonlat_to_healpix(l*u.deg, b*u.deg, self.maxnside, order='nested')
        i = self.sorter[np.searchsorted(self.index_nest, match_ipix, side='right', sorter=self.sorter) - 1]
        return self.ff['sigma_e%s'%self.version[-2:].lower()].values[i]

    def resolution(self, l, b):
        match_ipix = lonlat_to_healpix(l*u.deg, b*u.deg, self.maxnside, order='nested')
        i = self.sorter[np.searchsorted(self.index_nest, match_ipix, side='right', sorter=self.sorter) - 1]
        return self.ff['resolution'].values[i]
    
class extinction_map_healpix_glimpse(object):
    def __init__(self, zeropoint_correct=False):
        self.ff = pd.read_csv('/data/jls/virac/virac_extinction_map_healpix_glimpse.csv')
        self.level, self.index_nest = uniq_to_level_ipix(self.ff['uniq'])
        self.nside = level_to_nside(self.level[0])
        self.sorter = np.argsort(self.index_nest)
        
        if zeropoint_correct:
            zp = zeropoint_correcter('/data/jls/virac/zeropoints_v2.csv')
            lb=healpix_to_lonlat(self.index_nest, self.nside, order='nested')
            self.ff['eh45']-=zp.query(lb[0].deg,lb[1].deg,'H')
            
    def query(self, l, b):
        match_ipix = lonlat_to_healpix(l*u.deg, b*u.deg, self.nside, order='nested')
        i = self.sorter[np.searchsorted(self.index_nest, match_ipix, side='right', sorter=self.sorter) - 1]
        return self.ff['eh45'].values[i]
    def query_spread(self, l, b):
        match_ipix = lonlat_to_healpix(l*u.deg, b*u.deg, self.nside, order='nested')
        i = self.sorter[np.searchsorted(self.index_nest, match_ipix, side='right', sorter=self.sorter) - 1]
        return self.ff['sigma_eh45'].values[i]

#import atpy
 
def write_healpix_glimpse_wsdb():
    ffJK = pd.read_csv('/data/jls/virac/virac_extinction_map_healpix_glimpse.csv')
    ffJK['level'], ffJK['hpx'] = uniq_to_level_ipix(ffJK['uniq'])
    print(level_to_nside(np.unique(ffJK['level'])))
    t = atpy.Table()
    t.add_column('hpx', ffJK['hpx'])
#     t.add_column('l', ffJK['l'])
#     t.add_column('b', ffJK['b'])
    t.add_column('ak', ffJK['ak'])
    t.add_column('ak_spread', ffJK['sigma_ak'])
    t.table_name = 'vvv_glimpse_extinction_map_nest'
    t.write('postgres',
            user='jason_sanders',
            database='wsdb',
            host='cappc127.ast.cam.ac.uk',
            password=wsdbpassword,
            overwrite=True)
    
def write_healpix_wsdb(ffJK, version):
    ffJK['uniq_nest'] = level_ipix_to_uniq(ffJK['level'],ring2nest(level_to_nside(ffJK['level']),ffJK['ipix']))
    t = atpy.Table()
    t.add_column('uniq', ffJK['uniq_nest'])
    t.add_column('l', ffJK['l'])
    t.add_column('b', ffJK['b'])
    t.add_column('e%s'%version[-2:].lower(), ffJK['e%s'%version[-2:].lower()])
    t.add_column('e%s_spread'%version[-2:].lower(), ffJK['sigma_e%s'%version[-2:].lower()])
    t.add_column('counts', ffJK['counts'])
    t.add_column('resolution', ffJK['resolution'])
    lbl = 'healpix_moc_'
    t.table_name = 'vvv_%sextinction_map_%s' % (lbl, version)
    t.write('postgres',
            user='jason_sanders',
            database='wsdb',
            host='cappc127.ast.cam.ac.uk',
            password=wsdbpassword,
            overwrite=True)
    
def query_wsdb_extinction_map_healpix(l,b,version='jk'):
    hpx = lonlat_to_healpix(l*u.deg,b*u.deg,order='nested',nside=8192)
    map_query = \
           pd.DataFrame(sqlutil.local_join("""
                    select E.l, E.b, E.e{0}, E.e{0}_spread, resolution, counts from 
                    jason_sanders.vvv_healpix_moc_extinction_map_v2_{0} as E, 
                    mytable as t
                    where (t.hpx_     +268435456 = E.uniq or 
                           t.hpx_/4   +67108864  = E.uniq or 
                           t.hpx_/16  +16777216  = E.uniq or 
                           t.hpx_/64  +4194304   = E.uniq or 
                           t.hpx_/256 +1048576   = E.uniq or 
                           t.hpx_/1024+262144    = E.uniq)
                    """.format(version),
                    'mytable',(hpx,np.arange(len(hpx))),('hpx_','xid'),**wsdb_kwargs))
    return map_query

def build_wsdb_table(with_tmass=False):
    extM_jk = extinction_map('JK', add_tmass=with_tmass)
    extM_hk = extinction_map('HK', add_tmass=with_tmass)
    NSIDE = 2048  # 3 arcmin resolution
    import atpy, healpy as hp
    from login import wsdbpassword
    lrange = np.array([10. + 5. * with_tmass, 350. - 5. * with_tmass])
    brange = np.array([-10., 5. + 5. * with_tmass])
    centres = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    fltr = ((centres[0] < lrange[0]) | (centres[0] > lrange[1])) & (
        centres[1] < brange[1]) & (centres[1] > brange[0])
    l, b = centres[0][fltr], centres[1][fltr]
    EJK = extM_jk.query(l, b)
    sEJK = extM_jk.query_spread(l, b)
    EHK = extM_hk.query(l, b)
    sEHK = extM_hk.query_spread(l, b)
    healpix = np.arange(hp.nside2npix(NSIDE))[fltr]
    t = atpy.Table()
    t.add_column('hlpx', healpix)
    t.add_column('l', l)
    t.add_column('b', b)
    t.add_column('ejk', EJK)
    t.add_column('ejk_spread', sEJK)
    t.add_column('ehk', EHK)
    t.add_column('ehk_spread', sEHK)
    lbl = ''
    if with_tmass:
        lbl = 'tmass_'
    t.table_name = 'vvv_%sextinction_map_%s' % (lbl, version)
    t.write('postgres',
            user='jason_sanders',
            database='wsdb',
            host='cappc127.ast.cam.ac.uk',
            password=wsdbpassword,
            overwrite=True)


if __name__ == "__main__":
    #generate_healpix()
    #generate_healpix_NEW()
    build_extinction_map_healpix('v2_NEW_JK_PUBLIC','extmap_healpix_NEW.csv')
    build_extinction_map_healpix('v2_NEW_HK_PUBLIC','extmap_healpix_NEW.csv')
#     build_extinction_map_healpix('v2_NEW_HK','kawata/extmap_healpix_NEW.csv')
    #build_extinction_map_healpix('v2_HK','kawata/extmap_healpix.csv')
    #pass
    #build_extinction_map('v2_JK')
    #build_extinction_map('v2_HK')
    #build_disc_extinction_map('v2_JK')
    #build_disc_extinction_map('v2_HK')
#     build_2mass_extinction_map('v2_JK')
#     build_2mass_extinction_map('v2_HK')
#     build_wsdb_table()
#     build_wsdb_table(with_tmass=True)
#     write_healpix_glimpse_wsdb()
