import pandas as pd
import sqlutilpy as sqlutil
#from login import wsdbpassword
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
#from pm_transform import PM_Const

## GRAVITY experiment, uncertainty 0.03
R0 = 8.12
## Reid & Brunthaler (2004) Uncertainty 0.026
mulSgrA = -6.379
#Vsun = np.abs(R0 * mulSgrA * PM_Const)

# #wsdb_kwargs = {'host':'cappc127','user':'jason_sanders','password':wsdbpassword,
#                'asDict':True,'db':'wsdb',
#                'preamb': 'set enable_seqscan to off; set enable_mergejoin to off; '+
#                          'set enable_hashjoin to off;'}

def reflex_velocity(l, b, R00=8.12, deltaV=0.):
    ''' reflex motion in vl, vb and vlos '''
    vc = Vsun / R0 * R00
    vc += deltaV
    solar_peculiar = np.array([11.1, vc, 7.25])
    return solar_peculiar[0]*np.sin(l)-solar_peculiar[1]*np.cos(l),\
           solar_peculiar[0]*np.cos(l)*np.sin(b)+solar_peculiar[1]*np.sin(l)*np.sin(b)-solar_peculiar[2]*np.cos(b),\
           -solar_peculiar[0]*np.cos(l)*np.cos(b)-solar_peculiar[1]*np.sin(l)*np.cos(b)-solar_peculiar[2]*np.sin(b),


def grab_virac_wsdb_all(l, b, size, v2=False, v2_all=False, klimit=None):
    ''' 
        Grab the VIRAC data from the WSDB. 
    '''

    size /= 60.
    poly_string = "l>%0.3f and l<%0.3f and b>%0.3f and b<%0.3f"\
                    %(l-.5*size,l+.5*size,b-.5*size,b+.5*size)

    if (l - .5 * size < 0.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size+360.,l+.5*size,b-.5*size,b+.5*size)
    if (l + .5 * size > 360.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size,l+.5*size-360.,b-.5*size,b+.5*size)
    cat = 'leigh_smith.virac_pm_v11'
    kstring = 'mag'
    if v2:
        cat = 'leigh_smith.virac_pm2_jhk'
        kstring = 'kmag'
    if v2_all:
        cat = 'leigh_smith.virac_pm2'
        kstring = 'kmag'
    query = 'select * from %s ' % cat + 'where %s' % poly_string
    if klimit is not None:
        query = 'select * from %s ' % cat + 'where %s and %s<%0.4f' \
                        % (poly_string, kstring, klimit)
    v = pd.DataFrame(sqlutil.get(query,**wsdb_kwargs))
    return v


def grab_virac_wsdb(l, b, size, magerr=0.2):
    ''' 
        Grab the VIRAC data from the WSDB. Brightness cut at 11.5 mag. Cut on errors of magerr mag. 
    '''
    size /= 60.
    poly_string = "l>%0.3f and l<%0.3f and b>%0.3f and b<%0.3f"\
                    %(l-.5*size,l+.5*size,b-.5*size,b+.5*size)

    if (l - .5 * size < 0.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size+360.,l+.5*size,b-.5*size,b+.5*size)
    if (l + .5 * size > 360.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size,l+.5*size-360.,b-.5*size,b+.5*size)

    v = pd.DataFrame(
        sqlutil.get('select * from leigh_smith.virac_pm_v11 ' +
                    'where %s and mag>11.5 and mag<20 ' % poly_string +
                    'and emag<%0.3f' % magerr,**wsdb_kwargs))
    v.loc[v.jmag > 99., 'jmag'] = np.nan
    return v


def calibrate_wrt_2mass(l, b, size, virac):
    ''' size in arcmin '''
    size /= 60.
    poly_string = "glon>%0.3f and glon<%0.3f and glat>%0.3f and glat<%0.3f"\
                    %(l-.5*size,l+.5*size,b-.5*size,b+.5*size)

    if (l - .5 * size < 0.):
        poly_string = "(glon>%0.3f or glon<%0.3f) and glat>%0.3f and glat<%0.3f"\
                        %(l-.5*size+360.,l+.5*size,b-.5*size,b+.5*size)
    if (l + .5 * size > 360.):
        poly_string = "(glon>%0.3f or glon<%0.3f) and glat>%0.3f and glat<%0.3f"\
                        %(l-.5*size,l+.5*size-360.,b-.5*size,b+.5*size)

    tmass = pd.DataFrame(
        sqlutil.get("select * from twomass.psc " +
                    "where %s and ph_qual ~ '..[A-D]' and cc_flg like '__0'" %
                    poly_string,**wsdb_kwargs))
    c = SkyCoord(ra=virac.ra.values * u.degree, dec=virac.de.values * u.degree)
    catalog = SkyCoord(ra=tmass.ra.values * u.degree,
                       dec=tmass.decl.values * u.degree)
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    idx = idx[d2d.arcsec < .4]
    virac['dist'] = d2d.arcsec
    viracM = pd.concat([
        virac[d2d.arcsec < .4].reset_index(drop=True),
        tmass.loc[idx].reset_index(drop=True)
    ],
                       axis=1,
                       sort=False)
    viracM = viracM[np.abs(viracM.k_m - viracM.mag) < 1.].reset_index(
        drop=True)
    return viracM


def grab_2mass(l, b, size, qual_cut=True):
    ''' size in arcmin '''
    size /= 60.
    poly_string = "glon>%0.3f and glon<%0.3f and glat>%0.3f and glat<%0.3f"\
                    %(l-.5*size,l+.5*size,b-.5*size,b+.5*size)

    if (l - .5 * size < 0.):
        poly_string = "(glon>%0.3f or glon<%0.3f) and glat>%0.3f and glat<%0.3f"\
                        %(l-.5*size+360.,l+.5*size,b-.5*size,b+.5*size)
    if (l + .5 * size > 360.):
        poly_string = "(glon>%0.3f or glon<%0.3f) and glat>%0.3f and glat<%0.3f"\
                        %(l-.5*size,l+.5*size-360.,b-.5*size,b+.5*size)
    qstr = ""
    if qual_cut:
        qstr = "and ph_qual ~ '..[A-D]' and cc_flg like '___'"
    tmass = pd.DataFrame(
        sqlutil.get("select * from twomass.psc " +
                    "where %s %s and k_m>11 and k_m<20" % (poly_string, qstr),**wsdb_kwargs))
    tmass['mag'] = tmass['k_m'] + 0.01 * (tmass['j_m'] - tmass['k_m'])
    tmass['jmag'] = tmass['j_m'] - 0.065 * (tmass['j_m'] - tmass['k_m'])
    tmass['emag'] = tmass['k_msigcom']
    tmass['ejmag'] = tmass['j_msigcom']
    tmass['de'] = tmass['decl']
    tmass['l'] = tmass['glon']
    tmass['b'] = tmass['glat']
    tmass['rapm'] = np.nan
    tmass['depm'] = np.nan
    tmass['erapm'] = np.nan
    tmass['edepm'] = np.nan
    return tmass[[
        'ra', 'de', 'mag', 'emag', 'jmag', 'ejmag', 'l', 'b', 'rapm', 'depm',
        'erapm', 'edepm', 'j_m', 'k_m'
    ]]


def grab_wise(l, b, size):
    size /= 60.
    poly_string = "glon>%0.3f and glon<%0.3f and glat>%0.3f and glat<%0.3f"\
                    %(l-.5*size,l+.5*size,b-.5*size,b+.5*size)

    if (l - .5 * size < 0.):
        poly_string = "(glon>%0.3f or glon<%0.3f) and glat>%0.3f and glat<%0.3f"\
                        %(l-.5*size+360.,l+.5*size,b-.5*size,b+.5*size)
    if (l + .5 * size > 360.):
        poly_string = "(glon>%0.3f or glon<%0.3f) and glat>%0.3f and glat<%0.3f"\
                        %(l-.5*size,l+.5*size-360.,b-.5*size,b+.5*size)
    wise = pd.DataFrame(
        sqlutil.get("select * from allwise.main " +
                    "where %s and cc_flg like '00__'" % poly_string,**wsdb_kwargs))
    return wise

def cm_galacticnucleus(data, radeccols=['ra', 'de'], radius=1.):
    ra = data[radeccols[0]].values
    dec = data[radeccols[1]].values
    gnm = pd.DataFrame(
        sqlutil.local_join(
            """
		select *, q3c_dist(m.ra_,m.dec_,tt.ra,tt.dec) as dist from mytable as m
		left join lateral (select * from jason_sanders.galacticnucleus as s
		where q3c_join(m.ra_, m.dec_,s.ra,s.dec,%0.3f/3600)  
		order by q3c_dist(m.ra_,m.dec_,s.ra,s.dec) asc limit 1)
		as tt on  true  order by xid """ % radius,
            'mytable', (ra, dec, np.arange(len(dec))), ('ra_', 'dec_', 'xid'),
            host='cappc127',
            user='jason_sanders',
            password=wsdbpassword,
            asDict=True,
            preamb=
            'set enable_seqscan to off; set enable_mergejoin to off; set enable_hashjoin to off;',
            db='wsdb'
        ))
    return gnm

def cm_2mass(data, radeccols=['ra', 'de'], radius=1.):
    ra = data[radeccols[0]].values
    dec = data[radeccols[1]].values
    tmass = pd.DataFrame(
        sqlutil.local_join(
            """
		select *, q3c_dist(m.ra_,m.dec_,tt.ra,tt.decl) as dist from mytable as m
		left join lateral (select * from twomass.psc as s
		where q3c_join(m.ra_, m.dec_,s.ra,s.decl,%0.3f/3600)  
		order by q3c_dist(m.ra_,m.dec_,s.ra,s.decl) asc limit 1)
		as tt on  true  order by xid """ % radius,
            'mytable', (ra, dec, np.arange(len(dec))), ('ra_', 'dec_', 'xid'),
            host='cappc127',
            user='jason_sanders',
            password=wsdbpassword,
            asDict=True,
            preamb=
            'set enable_seqscan to off; set enable_mergejoin to off; set enable_hashjoin to off;',
            db='wsdb'
        ))
    return tmass

def cm_wise(data, radeccols=['ra', 'de'], radius=1.):
    ra = data[radeccols[0]].values
    dec = data[radeccols[1]].values
    wise = pd.DataFrame(
        sqlutil.local_join(
            """
		select *, q3c_dist(m.ra_,m.dec_,tt.ra,tt.dec) as dist from mytable as m
		left join lateral (select w1mpro,w2mpro,w1sigmpro,w2sigmpro,ph_qual,cc_flags,ra,dec from allwise.main as s
		where q3c_join(m.ra_, m.dec_,s.ra,s.dec,%0.3f/3600)  
		order by q3c_dist(m.ra_,m.dec_,s.ra,s.dec) asc limit 1)
		as tt on  true  order by xid """ % radius,
            'mytable', (ra, dec, np.arange(len(dec))), ('ra_', 'dec_', 'xid'),**wsdb_kwargs))
    wise['w1'] = wise['w1mpro']
    wise['w2'] = wise['w2mpro']
    wise['w1_err'] = wise['w1sigmpro']
    wise['w2_err'] = wise['w2sigmpro']
    return wise


def cm_unwise(data, radeccols=['ra', 'de'], withflags=True):
    ra = data[radeccols[0]].values
    dec = data[radeccols[1]].values
    flags = "and s.flags_unwise_w1=0 and s.flags_unwise_w2=0"
    if not withflags:
        flags = ""
    wise = pd.DataFrame(
        sqlutil.local_join(
            """
		select * from mytable as m
		left join lateral (select flux_w1,flux_w2,dflux_w1,dflux_w2,flags_unwise_w1,flags_unwise_w2,ra,dec from unwise_1901.main as s
		where q3c_join(m.ra_, m.dec_,s.ra,s.dec,1./3600) %s 
		order by q3c_dist(m.ra_,m.dec_,s.ra,s.dec) asc limit 1)
		as tt on  true  order by xid """ % flags,
            'mytable', (ra, dec, np.arange(len(dec))), ('ra_', 'dec_', 'xid'),**wsdb_kwargs))
    wise['w1'] = 22.5 - 2.5 * np.log10(wise['flux_w1'])
    wise['w2'] = 22.5 - 2.5 * np.log10(wise['flux_w2'])
    wise['w1_err'] = 2.5 / np.log(10.) * wise['dflux_w1'] / wise['flux_w1']
    wise['w2_err'] = 2.5 / np.log(10.) * wise['dflux_w2'] / wise['flux_w2']
    return wise


def cm_gaia(data, radeccols=['ra', 'de'], cm_radius=1.):
    ra = data[radeccols[0]].values
    dec = data[radeccols[1]].values
    decaps = pd.DataFrame(
        sqlutil.local_join(
            """
                select * from mytable as m
                left join lateral (select *, q3c_dist(m.ra,m.dec,s.ra,s.dec) from gaia_dr2.gaia_source as s
                where q3c_join(m.ra, m.dec,s.ra,s.dec,%0.4f/3600) 
                order by q3c_dist(m.ra,m.dec,s.ra,s.dec) asc limit 1)
                as tt on  true  order by xid """ % cm_radius,
            'mytable', (ra, dec, np.arange(len(dec))), ('ra', 'dec', 'xid'),**wsdb_kwargs))
    return decaps


def cm_virac(data, radeccols=['ra', 'de'], cm_radius=1., catalog='v1'):
    ra = data[radeccols[0]].values
    dec = data[radeccols[1]].values
    cat = 'leigh_smith.virac_pm_v11'
    decfld = 'de'
    if catalog == 'v2':
        cat = 'leigh_smith.virac_pm2_jhk'
        decfld = 'dec'
    if catalog == 'v2_full':
        cat = 'leigh_smith.virac_pm2'
        decfld = 'dec'
    decaps = \
       sqlutil.local_join("""
                select * from mytable as m
                left join lateral (select *, q3c_dist(m.ra_,m.dec_,s.ra,s.{1}) from {0} as s
                where q3c_join(m.ra_, m.dec_,s.ra,s.{1},{2}/3600) 
                order by q3c_dist(m.ra_,m.dec_,s.ra,s.{1}) asc limit 1)
                as tt on  true  order by xid """.format(cat,decfld,cm_radius),
                'mytable',(ra,dec,np.arange(len(dec))),('ra_','dec_','xid'),**wsdb_kwargs)
    decaps = pd.DataFrame(decaps)
    return decaps

def cm_virac_KMAG(data, radecKcols=['ra', 'de', 'Kmag'], cm_radius=1., catalog='v1', kmag_radius=1.):
    ''' Similar to above but also match on K'''
    ra = data[radecKcols[0]].values
    dec = data[radecKcols[1]].values
    kmag = data[radecKcols[2]].values
    cat = 'leigh_smith.virac_pm_v11'
    decfld = 'de'
    kmagfld = 'mag'
    if catalog == 'v2':
        cat = 'leigh_smith.virac_pm2_jhk'
        decfld = 'dec'
        kmagfld = 'kmag'
    if catalog == 'v2_full':
        cat = 'leigh_smith.virac_pm2'
        decfld = 'dec'
        kmagfld = 'kmag'
    decaps = \
       sqlutil.local_join("""
                select * from mytable as m
                left join lateral (select *, q3c_dist(m.ra_,m.dec_,s.ra,s.{1}) from {0} as s
                where q3c_join(m.ra_, m.dec_,s.ra,s.{1},{2}/3600) and abs(m.kmag_-s.{3})<{4}
                order by q3c_dist(m.ra_,m.dec_,s.ra,s.{1}) asc limit 1)
                as tt on  true  order by xid """.format(cat,decfld,cm_radius,kmagfld,kmag_radius),
                'mytable',(ra,dec,kmag,np.arange(len(dec))),('ra_','dec_','kmag_','xid'),
                          **wsdb_kwargs)
    decaps = pd.DataFrame(decaps)
    return decaps


def cm_decaps(data, radeccols=['ra', 'de']):
    ra = data[radeccols[0]].values
    dec = data[radeccols[1]].values
    decaps = pd.DataFrame(
        sqlutil.local_join(
            """
		select * from mytable as m
		left join lateral (select * from decaps_dr1.main as s
		where q3c_join(m.ra, m.dec,s.ra,s.dec,1./3600) 
		order by q3c_dist(m.ra,m.dec,s.ra,s.dec) asc limit 1)
		as tt on  true  order by xid """,
            'mytable', (ra, dec, np.arange(len(dec))), ('ra', 'dec', 'xid'),**wsdb_kwargs))
    return decaps


def cm_apass(data, radeccols=['ra', 'de']):
    ra = data[radeccols[0]].values
    dec = data[radeccols[1]].values
    decaps = pd.DataFrame(
        sqlutil.local_join(
            """
		select * from mytable as m
		left join lateral (select * from apassdr9.main as s
		where q3c_join(m.ra, m.dec,s.ra,s.dec,1./3600) 
		order by q3c_dist(m.ra,m.dec,s.ra,s.dec) asc limit 1)
		as tt on  true  order by xid """,
            'mytable', (ra, dec, np.arange(len(dec))), ('ra', 'dec', 'xid'),**wsdb_kwargs))
    return decaps


def cm_ps1(data, radeccols=['ra', 'de']):
    ra = data[radeccols[0]].values
    dec = data[radeccols[1]].values
    decaps = pd.DataFrame(
        sqlutil.local_join(
            """
		select * from mytable as m
		left join lateral (select * from panstarrs_dr1.stackobjectthin as s
		where q3c_join(m.ra, m.dec,s.ra,s.dec,1./3600) 
		order by q3c_dist(m.ra,m.dec,s.ra,s.dec) asc limit 1)
		as tt on  true  order by xid """,
            'mytable', (ra, dec, np.arange(len(dec))), ('ra', 'dec', 'xid'),**wsdb_kwargs))
    return decaps


def cm_glimpse(data, radeccols=['ra', 'de'], cols=None):
    ra = data[radeccols[0]].values
    dec = data[radeccols[1]].values
    select_cols = "*"
    if cols is not None:
        select_cols=",".join(["tt."+c for c in cols])
    gc3 = pd.DataFrame(
        sqlutil.local_join(
            """
		select %s from mytable as m
		left join lateral (select * from glimpse.catalog3 as s
		where q3c_join(m.ra_, m.dec_,s.ra,s.dec,1./3600)  
		order by q3c_dist(m.ra_,m.dec_,s.ra,s.dec) asc limit 1)
		as tt on  true  order by xid """%select_cols,
            'mytable', (ra, dec, np.arange(len(dec))), ('ra_', 'dec_', 'xid'),**wsdb_kwargs))
    gc2 = pd.DataFrame(
        sqlutil.local_join(
            """
		select %s from mytable as m
		left join lateral (select * from glimpse.catalog2 as s
		where q3c_join(m.ra_, m.dec_,s.ra,s.dec,1./3600)  
		order by q3c_dist(m.ra_,m.dec_,s.ra,s.dec) asc limit 1)
		as tt on  true  order by xid """%select_cols,
            'mytable', (ra, dec, np.arange(len(dec))), ('ra_', 'dec_', 'xid'),**wsdb_kwargs))
    fltr = (gc3['mag3_6'] != gc3['mag3_6'])
    gc3[fltr] = gc2[fltr]
    return gc3


def grab_gaia(l, b, size):
    ''' size in arcmin '''
    size /= 60.
    poly_string = "l>%0.3f and l<%0.3f and b>%0.3f and b<%0.3f"\
                    %(l-.5*size,l+.5*size,b-.5*size,b+.5*size)
    if (l - .5 * size < 0.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size+360.,l+.5*size,b-.5*size,b+.5*size)
    if (l + .5 * size > 360.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size,l+.5*size-360.,b-.5*size,b+.5*size)

    gaia = pd.DataFrame(
        sqlutil.get("select * from gaia_dr2.gaia_source " +
                    "where %s" % poly_string,**wsdb_kwargs))
    return gaia


def grab_gaia_and_crossmatch_wsdb(l, b, size, virac):
    ''' size in arcmin '''
    size /= 60.
    poly_string = "l>%0.3f and l<%0.3f and b>%0.3f and b<%0.3f"\
                    %(l-.5*size,l+.5*size,b-.5*size,b+.5*size)
    if (l - .5 * size < 0.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size+360.,l+.5*size,b-.5*size,b+.5*size)
    if (l + .5 * size > 360.):
        poly_string = "(l>%0.3f or l<%0.3f) and b>%0.3f and b<%0.3f"\
                        %(l-.5*size,l+.5*size-360.,b-.5*size,b+.5*size)

    gaia = pd.DataFrame(
        sqlutil.get("select * from gaia_dr2.gaia_source " +
                    "where %s" % poly_string,**wsdb_kwargs))
    c = SkyCoord(ra=virac.ra.values * u.degree, dec=virac.de.values * u.degree)
    catalog = SkyCoord(ra=gaia.ra.values * u.degree,
                       dec=gaia.dec.values * u.degree)
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    idx = idx[d2d.arcsec < 1.]
    virac['dist'] = d2d.arcsec
    del gaia['b']
    del gaia['l']
    del gaia['ra']
    del gaia['dec']
    viracM = pd.concat([
        virac[d2d.arcsec < 1.].reset_index(drop=True),
        gaia.loc[idx].reset_index(drop=True)
    ],
                       axis=1,
                       sort=False)
    virac_app = virac[d2d.arcsec>1.].reset_index(drop=True)
    # This forces source_id to be int64 -- useful for removing duplciate crossmatches (although only a ~0.1% effect)
    #virac_app['source_id']=-99
    viracM = viracM.append(virac_app).reset_index(drop=True)
    #print(viracM['source_id'].dtype)
    viracM['gaia_pmra_pmdec_corr'] = viracM['pmra_pmdec_corr']
    ## Inverse variance weighting
    ## pmra = (gaia_pmra/gaia_pmra_error**2+virac_pmra/virac_pmra_error**2)/(1./gaia_pmra_error**2+1./virac_pmra_error**2)
    ## cov(pmra,pmdec) = cov(gaia_pmra,gaia_pmdec)/gaia_pmra_error**2/gaia_pmdec_error**2 / (1./gaia_pmra_error**2+1./virac_pmra_error**2) / (1./gaia_pmdec_error**2+1./virac_pmdec_error**2) = corr /gaia_pmra_error/gaia_pmdec_error / (1./gaia_pmra_error**2+1./virac_pmra_error**2) / (1./gaia_pmdec_error**2+1./virac_pmdec_error**2) --- no correlations in VIRAC
    viracM.loc[viracM['pmra'] == viracM['pmra'], 'rapm'] = (
        (viracM['rapm'] / viracM['erapm']**2 +
         viracM['pmra'] / viracM['pmra_error']**2) /
        (1. / viracM['erapm']**2 +
         1. / viracM['pmra_error']**2))[viracM['pmra'] == viracM['pmra']]
    viracM.loc[viracM['pmdec'] == viracM['pmdec'], 'depm'] = (
        (viracM['depm'] / viracM['edepm']**2 +
         viracM['pmdec'] / viracM['pmdec_error']**2) /
        (1. / viracM['edepm']**2 +
         1. / viracM['pmdec_error']**2))[viracM['pmdec'] == viracM['pmdec']]
    ## First compute covariance -- then divide by errors later to find correlation
    viracM.loc[viracM['pmra_pmdec_corr'] ==
               viracM['pmra_pmdec_corr'], 'pmra_pmdec_corr'] = (
                   viracM['pmra_pmdec_corr'] / viracM['pmra_error'] /
                   viracM['pmdec_error'] /
                   (1. / viracM['erapm']**2 + 1. / viracM['pmra_error']**2) /
                   (1. / viracM['edepm']**2 + 1. / viracM['pmdec_error']**2)
               )[viracM['pmra_pmdec_corr'] == viracM['pmra_pmdec_corr']]
    viracM.loc[viracM['pmra_pmdec_corr'] != viracM['pmra_pmdec_corr'],
               'pmra_pmdec_corr'] = 0.
    viracM.loc[viracM['pmra_error'] ==
               viracM['pmra_error'], 'erapm'] = 1. / np.sqrt(
                   1. / viracM['erapm']**2 + 1. / viracM['pmra_error']**2)[
                       viracM['pmra_error'] == viracM['pmra_error']]
    viracM.loc[viracM['pmdec_error'] ==
               viracM['pmdec_error'], 'edepm'] = 1. / np.sqrt(
                   1. / viracM['edepm']**2 + 1. / viracM['pmdec_error']**2)[
                       viracM['pmdec_error'] == viracM['pmdec_error']]
    viracM['pmra_pmdec_corr'] = viracM['pmra_pmdec_corr'] / viracM[
        'erapm'] / viracM['edepm']

    fltr = (viracM['rapm'] != viracM['rapm'])
    viracM.loc[fltr, 'rapm'] = viracM['pmra'][fltr]
    viracM.loc[fltr, 'depm'] = viracM['pmdec'][fltr]
    viracM.loc[fltr, 'erapm'] = viracM['pmra_error'][fltr]
    viracM.loc[fltr, 'edepm'] = viracM['pmdec_error'][fltr]

    return viracM


def grab_virac_wsdb_extinction_cut(l, b, size, kmax):
    ''' 
        Grab the VIRAC data from the WSDB. Brightness cut at 11.5 mag. Cut on errors of 0.2mag. 
        Only consider stars with j-k>0.4
        Only grab stars with (k<kmax+(j-k-0.62)*akconst)&(j-k>0.5)
    '''

    akconst = 0.482
    size /= 60.
    poly_string = "l>%0.3f and l<%0.3f and b>%0.3f and b<%0.3f"\
                    %(l-.5*size,l+.5*size,b-.5*size,b+.5*size)

    v = pd.DataFrame(
        sqlutil.get(
            'select * from leigh_smith.virac_pm_v11 ' +
            'where %s and mag>11.5 and mag<20 and jmag>0 and jmag<20 ' %
            poly_string +
            'and ejmag<0.2 and emag<0.2 and jmag-mag>0.5 and (mag<%0.3f+(jmag-mag-0.62)*0.482)'
            % kmax,**wsdb_kwargs))
    return v


def grab_2mass_wsdb_extinction_cut(l, b, size, kmax):
    ''' 
        Grab the 2MASS data from the WSDB. Brightness cut at 11.5 mag. Cut on errors of 0.2mag. 
        Only consider stars with j-k>0.4
        Only grab stars with (k<kmax+(j-k-0.62)*akconst)&(j-k>0.5)
    '''

    akconst = 0.482
    size /= 60.
    poly_string = "glon>%0.3f and glon<%0.3f and glat>%0.3f and glat<%0.3f"\
                    %(l-.5*size,l+.5*size,b-.5*size,b+.5*size)

    tmass = pd.DataFrame(
        sqlutil.get(
            "select * from twomass.psc " +
            "where %s and ph_qual ~ '[A-D].[A-D]' and cc_flg like '___' " %
            poly_string + 'and k_m>11.5 and k_m<20 and j_m>0 and j_m<20 ' +
            'and j_msigcom<0.2 and k_msigcom<0.2 and j_m-k_m>0.5 and '
            '(k_m<%0.3f+(0.925*(j_m-k_m)-0.62)*0.482)' % kmax,**wsdb_kwargs))
    tmass['mag'] = tmass['k_m'] + 0.01 * (tmass['j_m'] - tmass['k_m'])
    tmass['jmag'] = tmass['j_m'] - 0.065 * (tmass['j_m'] - tmass['k_m'])
    tmass['emag'] = tmass['k_msigcom']
    tmass['ejmag'] = tmass['j_msigcom']
    tmass['de'] = tmass['decl']
    tmass['l'] = tmass['glon']
    tmass['b'] = tmass['glat']
    return tmass


def crossmatch_virac_gaia(ra, dec):
    """
        Cross match list of ra,dec of stars with VIRAC and Gaia -- 1 arcsec
    """
    data = pd.DataFrame(
        sqlutil.local_join(
            """
        select * from mytable as m
        left join lateral (select * from leigh_smith.virac_pm_v11 as s
        where q3c_join(m.ra, m.dec,s.ra,s.de,1./3600)
        order by q3c_dist(m.ra,m.dec,s.ra,s.de) asc limit 1)
        as tt on  true  order by xid """,
            'mytable', (ra, dec, np.arange(len(dec))), ('ra', 'dec', 'xid'),**wsdb_kwargs))
    del data['ra']
    del data['dec']
    gaia = pd.DataFrame(
        sqlutil.local_join(
            """
        select * from mytable as m
        left join lateral (select * from gaia_dr2.gaia_source as s
        where q3c_join(m.ra, m.dec,s.ra,s.dec,1./3600)
        order by q3c_dist(m.ra,m.dec,s.ra,s.dec) asc limit 1)
        as tt on  true  order by xid """,
            'mytable', (ra, dec, np.arange(len(dec))), ('ra', 'dec', 'xid'),**wsdb_kwargs))
    del gaia['b']
    del gaia['l']
    del gaia['ra']
    del gaia['dec']
    viracM = pd.concat([data, gaia], axis=1, sort=False)
    viracM['gaia_pmra_pmdec_corr'] = viracM['pmra_pmdec_corr']
    ## Inverse variance weighting
    ## pmra = (gaia_pmra/gaia_pmra_error**2+virac_pmra/virac_pmra_error**2)/(1./gaia_pmra_error**2+1./virac_pmra_error**2)
    ## cov(pmra,pmdec) = cov(gaia_pmra,gaia_pmdec)/gaia_pmra_error**2/gaia_pmdec_error**2 / (1./gaia_pmra_error**2+1./virac_pmra_error**2) / (1./gaia_pmdec_error**2+1./virac_pmdec_error**2) = corr /gaia_pmra_error/gaia_pmdec_error / (1./gaia_pmra_error**2+1./virac_pmra_error**2) / (1./gaia_pmdec_error**2+1./virac_pmdec_error**2) --- no correlations in VIRAC
    viracM.loc[viracM['pmra'] == viracM['pmra'], 'rapm'] = (
        (viracM['rapm'] / viracM['erapm']**2 +
         viracM['pmra'] / viracM['pmra_error']**2) /
        (1. / viracM['erapm']**2 +
         1. / viracM['pmra_error']**2))[viracM['pmra'] == viracM['pmra']]
    viracM.loc[viracM['pmdec'] == viracM['pmdec'], 'depm'] = (
        (viracM['depm'] / viracM['edepm']**2 +
         viracM['pmdec'] / viracM['pmdec_error']**2) /
        (1. / viracM['edepm']**2 +
         1. / viracM['pmdec_error']**2))[viracM['pmdec'] == viracM['pmdec']]
    ## First compute covariance -- then divide by errors later to find correlation
    viracM.loc[viracM['pmra_pmdec_corr'] ==
               viracM['pmra_pmdec_corr'], 'pmra_pmdec_corr'] = (
                   viracM['pmra_pmdec_corr'] / viracM['pmra_error'] /
                   viracM['pmdec_error'] /
                   (1. / viracM['erapm']**2 + 1. / viracM['pmra_error']**2) /
                   (1. / viracM['edepm']**2 + 1. / viracM['pmdec_error']**2)
               )[viracM['pmra_pmdec_corr'] == viracM['pmra_pmdec_corr']]
    viracM.loc[viracM['pmra_pmdec_corr'] != viracM['pmra_pmdec_corr'],
               'pmra_pmdec_corr'] = 0.
    viracM.loc[viracM['pmra_error'] ==
               viracM['pmra_error'], 'erapm'] = 1. / np.sqrt(
                   1. / viracM['erapm']**2 + 1. / viracM['pmra_error']**2)[
                       viracM['pmra_error'] == viracM['pmra_error']]
    viracM.loc[viracM['pmdec_error'] ==
               viracM['pmdec_error'], 'edepm'] = 1. / np.sqrt(
                   1. / viracM['edepm']**2 + 1. / viracM['pmdec_error']**2)[
                       viracM['pmdec_error'] == viracM['pmdec_error']]
    viracM['pmra_pmdec_corr'] = viracM['pmra_pmdec_corr'] / viracM[
        'erapm'] / viracM['edepm']

    fltr = (viracM['rapm'] != viracM['rapm'])
    viracM.loc[fltr, 'rapm'] = viracM['pmra'][fltr]
    viracM.loc[fltr, 'depm'] = viracM['pmdec'][fltr]
    viracM.loc[fltr, 'erapm'] = viracM['pmra_error'][fltr]
    viracM.loc[fltr, 'edepm'] = viracM['pmdec_error'][fltr]

    return viracM
