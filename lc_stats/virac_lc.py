import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlutilpy as sqlutil
from astropy import time, coordinates as coord, units as u
from astropy.coordinates import SkyCoord

from lc_utils import wrap_df
from wsdb_cred import wsdb_kwargs
#from gaussian_process import *

paranal = coord.EarthLocation.of_site('paranal')


def load_virac_lc(virac_id, ver='1', band='K_s', extra_info=None):
    ''' If coordinate, correct MJD to HJD '''
    if band == 'K_s':
        band = ''
        if ver=='2':
            cat = 'virac2_ts_tmolnar'
        elif ver=='1':
            cat = 'virac_lc'
    else:
        band = band.lower() + '_'
        cat = 'virac2_zyjh_lc'
    
    if ver=='2':
        query = """
           select unnest({0}mjdobs) as mjdobs, 
               unnest({0}mag) as mag, 
               unnest({0}emag) as error,
               unnest({0}chi) as chi,
               unnest(ast_res_chisq) as ast_res_chisq,
               unnest(ambiguous_match) as ambiguous_match,
               unnest(filterid)
           from leigh_smith.{1}
           where sourceid={2}
           """.format(band, cat, virac_id)
    
    elif ver=='1':
        query = """
               select unnest({0}mjdobs) as mjdobs, 
                   unnest({0}mag) as mag, 
                   unnest({0}emag) as error,
                   unnest({0}dp_chi) as dpchi
               from leigh_smith.{1}
               where sourceid={2}""".format(band, cat, virac_id)
    
    if extra_info is not None:
        if extra_info==True:
            einfo="v.*"
        else:
            einfo = ",".join(["v.%s"%s for s in extra_info])
        extra_lc_info="""
               unnest(ext) as ext, 
               unnest(dp_objtype) as dp_objtype,
               unnest(pxl_cnf) as pxl_cnf, 
               unnest(sky) as sky,
               unnest(x) as x, unnest(y) as y,
               unnest(detid) as detid,
            """
        query = """
           with g as (select unnest({0}mjdobs) as mjdobs, 
               unnest({0}mag) as mag, 
               unnest({0}emag) as error,
               unnest({0}dp_chi) as dpchi, 
               unnest(catid) as catid
           from leigh_smith.{1} where sourceid={2})
           select g.*, {3} from g, leigh_smith.virac2_cat_index as v 
           where g.catid=v.id
           """.format(band, cat, virac_id, einfo)
        
        
        query = """
           with g as (select unnest({0}mjdobs) as mjdobs, 
               unnest({0}mag) as mag, 
               unnest({0}emag) as error,
               unnest({0}dp_chi) as dpchi, 
               unnest(catid) as catid
           from leigh_smith.{1} where sourceid={2})
           select g.*, {3} from g inner join leigh_smith.virac2_cat_index as v 
           on g.catid=v.id
           """.format(band, cat, virac_id, einfo)
        
    data = pd.DataFrame(sqlutil.get(query,**wsdb_kwargs))
    
    if len(data) == 0:
        return data
    
    if ver=='2':
        radecdata = pd.DataFrame(
        sqlutil.get("""
                    select ra,dec from leigh_smith.virac2_ts_tmolnar
                    where sourceid={0}""".format(virac_id),
                   **wsdb_kwargs))
    elif ver=='1':
        radecdata = pd.DataFrame(
            sqlutil.get("""
                        select ra,dec from leigh_smith.virac_pm2 
                        where sourceid={0}""".format(virac_id),
                       **wsdb_kwargs))
        
    coordinate = SkyCoord(ra=radecdata['ra'].values[0] * u.deg,
                          dec=radecdata['dec'].values[0] * u.deg,
                          frame='icrs')
    times = time.Time(data['mjdobs'],
                      format='mjd',
                      scale='utc',
                      location=paranal)
    
    data['HJD'] = data['mjdobs'] + 2400000.5 + times.light_travel_time(
        coordinate).to(u.day).value

    return data


def find_nearest_virac(ra, dec, radius=1. / 3600.):
    data = pd.DataFrame(
        sqlutil.get("""
        select sourceid,kmag from leigh_smith.virac_pm2_jhk where q3c_radial_query(ra,dec,{0},{1},{2}) 
        order by q3c_dist(ra,dec,{0},{1}) asc limit 1""".format(
            ra, dec, radius),
                   **wsdb_kwargs))
    if len(data) > 0:
        return data['sourceid'].values[0]
    else:
        return None


def plot_virac_lc(virac_id, band='K_s', period=None, color='r'):
    if virac_id is None:
        return
    fld = 'HJD'
    data = load_virac_lc(virac_id, band=band)
    data = wrap_df(data, fld, period)
    plt.errorbar(data[fld],
                 data['mag'],
                 yerr=data['error'],
                 fmt='o',
                 ms=5,
                 label=r'$%s$ mag' % band,
                 color=color)
    plt.xlabel(fld)
    plt.ylabel('Magnitude')
    #     plt.gca().invert_yaxis()
    return None

def load_virac_lc_idlist(sourceids, band='K_s', kmax=200., kepoch_cut=0.):
    '''
    Get all light curves within a specified list of sourceids
    ----
    returns: list of panda Dataframes each stroing a single source lightcurve
    
    '''
    if band == 'K_s':
        band = ''
        cat = 'virac_lc'
    else:
        band = band.lower() + '_'
        cat = 'virac2_zyjh_lc'
        
    query = """with x as (select * from leigh_smith.virac_pm2 where
                          sourceid in {2}
                          and kmag<{3} and kepochs>{4})
               select x.ra,x.dec,l.sourceid,unnest({0}mjdobs) as mjdobs, 
                   unnest({0}mag) as mag, 
                   unnest({0}emag) as error,
                   unnest({0}dp_chi) as dpchi
               from leigh_smith.{1} as l, x
               where l.sourceid=x.sourceid;
          """.format(band, cat, 
                     tuple(sourceids),
                     kmax, kepoch_cut)
    
    data = sqlutil.get(query, **wsdb_kwargs)
    
    # Correct MJD to HJD
    coordinate = SkyCoord(ra=data['ra'] * u.deg,
                          dec=data['dec'] * u.deg,
                          frame='icrs')
    times = time.Time(data['mjdobs'][0],
                      format='mjd',
                      scale='utc',
                      location=paranal)
    data['HJD'] = data['mjdobs'] + 2400000.5 + times.light_travel_time(
        coordinate).to(u.day).value
    
    
    # Now split arrays into list of pandas dataframes
    indices = np.argwhere(np.diff(data['sourceid']) != 0).flatten() + 1
    datadict = {c: np.split(data[c], indices) for c in list(data.keys())}
    lightcurves = [ pd.DataFrame(dict(list(zip(datadict, t))))
                   for t in zip(*list(datadict.values()))
                  ]
    
    return lightcurves
    

def load_virac_lc_lbbox(l, b, size, band='K_s', kmax=200., kepoch_cut=0., extra_info=None):
    ''' Get all light curves within a box of size (in arcmin) centred on l, b '''

    if l<0.:
        l+=360.
    
    if band == 'K_s':
        band = ''
        cat = 'virac_lc'
    else:
        band = band.lower() + '_'
        cat = 'virac2_zyjh_lc'
    query = """with x as (select * from leigh_smith.virac_pm2 where
                          l>{2} and l<{3} and b>{4} and b<{5}
                          and kmag<{6} and kepochs>{7})
               select x.ra,x.dec,l.sourceid,unnest({0}mjdobs) as mjdobs, 
                   unnest({0}mag) as mag, 
                   unnest({0}emag) as error,
                   unnest({0}dp_chi) as dpchi
               from leigh_smith.{1} as l, x
               where l.sourceid=x.sourceid;
          """.format(band, cat, 
                     l - size / 60. / 2., l + size / 60. / 2.,
                     b - size / 60. / 2., b + size / 60. / 2.,
                     kmax, kepoch_cut)
    if extra_info is not None:
        if extra_info==True:
            einfo="v.*"
        else:
            einfo = ",".join(["v.%s"%s for s in extra_info])
        extra_info_lc = """
                       unnest(ext) as ext, unnest(dp_objtype) as dp_objtype,
                       unnest(pxl_cnf) as pxl_cnf, unnest(sky) as sky,
                       unnest(x) as x, unnest(y) as y,
                       unnest(detid) as detid, 
                       """
        query = """with y as(
                  with x as (select * from leigh_smith.virac_pm2 where
                              l>{2} and l<{3} and b>{4} and b<{5}
                              and kmag<{6} and kepochs>{7})
                   select x.ra,x.dec,l.sourceid,unnest({0}mjdobs) as mjdobs, 
                       unnest({0}mag) as mag, 
                       unnest({0}emag) as error,
                       unnest({0}dp_chi) as dpchi,
                       unnest(catid) as catid
                   from leigh_smith.{1} as l, x
                   where l.sourceid=x.sourceid)
                   select y.*, {8} from y inner join
                   leigh_smith.virac2_cat_index as v 
                   on y.catid=v.id;
              """.format(band, cat, 
                         l - size / 60. / 2., l + size / 60. / 2.,
                         b - size / 60. / 2., b + size / 60. / 2.,
                         kmax, kepoch_cut, einfo)
    data = sqlutil.get(query, **wsdb_kwargs)
    ## Correct MJD to HJD
    coordinate = SkyCoord(ra=data['ra'] * u.deg,
                          dec=data['dec'] * u.deg,
                          frame='icrs')
    times = time.Time(data['mjdobs'][0],
                      format='mjd',
                      scale='utc',
                      location=paranal)
    data['HJD'] = data['mjdobs'] + 2400000.5 + times.light_travel_time(
        coordinate).to(u.day).value

    # Now split arrays into list of pandas dataframes
    indices = np.argwhere(np.diff(data['sourceid']) != 0).flatten() + 1
    datadict = {c: np.split(data[c], indices) for c in list(data.keys())}
    lightcurves = [
        pd.DataFrame(dict(list(zip(datadict, t))))
        for t in zip(*list(datadict.values()))
    ]

    query = """select * from leigh_smith.virac_pm2 where
               l>{2} and l<{3} and b>{4} and b<{5}  
               and kmag<{6} and kepochs>{7}
          """.format(band, cat, 
                     l - size / 60. / 2., l + size / 60. / 2.,
                     b - size / 60. / 2., b + size / 60. / 2.,
                     kmax, kepoch_cut)
    data = pd.DataFrame(
        sqlutil.get(query, **wsdb_kwargs))

    return lightcurves, data


xnorm3 = 0.021
xcut3 = 12.


def quality_cut_phot(data, ver='1', strict_factor=1., dpchicut=5.):
    #     data = data[data['error']<1./strict_factor*(
    #                  xnorm3 * (1. + 10.**(0.4*(data['mag']-xcut3)))+xnorm3)].reset_index(drop=True)
    maglimit = 13.2 ## BASED ON EXPERIMENTS WITH MATSUNAGA
    if ver=='2':
        data = data[~((data['chi'] > dpchicut) &
                  (data['mag'] < maglimit))].reset_index(drop=True)
    elif ver=='1':
        data = data[~((data['dpchi'] > dpchicut) &
                      (data['mag'] < maglimit))].reset_index(drop=True)
    return data


def sigclipper(data, thresh=5.):
    if len(data) <= 1:
        return data
    stdd = .5 * np.diff(np.nanpercentile(data['mag'].values, [16, 84]))
    midd = np.nanmedian(data['mag'].values)
    return data[np.abs(data['mag'].values - midd) / stdd < thresh].reset_index(
        drop=True)

def load_quality_virac_lc(virac_id,thresh=5.,ver='1',**kwargs):
    return sigclipper(quality_cut_phot(load_virac_lc(virac_id, ver=ver, **kwargs), ver=ver), thresh=thresh)

def test_lc(virac_id,amb_corr=True,none=True,**kwargs):
    
    if none:
        return load_virac_lc(virac_id,amb_corr,**kwargs)
    else:
        return quality_cut_phot(load_virac_lc(virac_id,amb_corr,**kwargs),amb_corr)


### Correcting light-curves

def correct_nearby(vG, sid, radec=None, with_plot=True, log=False, cmradius=1., fld='seeing'):

    vvvG = vG.copy()
    
    if radec is None:
        queryradec = """select ra, dec from leigh_smith.virac_pm2 
                where sourceid={0}""".format(sid)
        dataradec = pd.DataFrame(sqlutil.get(queryradec, **wsdb_kwargs))
        ra, dec = dataradec['ra'].values[0], dataradec['dec'].values[0]
    else:
        ra, dec = radec[0], radec[1]
    
    ## Grab nearby data
    query2 = """select x.*, q3c_dist(x.ra,x.dec,{0},{1})*3600. as q3c_d 
                from leigh_smith.virac_pm2 as x 
                where q3c_radial_query(x.ra,x.dec,{0},{1},{2}/3600.)""".format(ra,dec,cmradius)
    nearbydata = pd.DataFrame(sqlutil.get(query2, **wsdb_kwargs))
    nearbydata = nearbydata.sort_values(by='q3c_d').reset_index(drop=True).reset_index(drop=True)
    
    ## Only reliable stuff
    
    OR_ = True
    if OR_:
        nearbyfltr = (nearbydata['bestconsecdets'].values[1:]>=5)|(nearbydata['pp2frac'].values[1:]>0.2)
    else:
        nearbyfltr = (nearbydata['bestconsecdets'].values[1:]>=5)&(nearbydata['pp2frac'].values[1:]>0.2)
    JHOR_ = True
    if JHOR_:
        nearbyfltr |= (nearbydata['jmag'].values[1:]==nearbydata['jmag'].values[1:]) | \
                      (nearbydata['hmag'].values[1:]==nearbydata['hmag'].values[1:])
    else:
        nearbyfltr |= (nearbydata['jmag'].values[1:]==nearbydata['jmag'].values[1:]) & \
                      (nearbydata['hmag'].values[1:]==nearbydata['hmag'].values[1:])
    NB=np.count_nonzero(nearbyfltr)
    NB+=1*(NB<0)
    nearbydata = pd.concat((nearbydata[:1],nearbydata[1:][nearbyfltr])).reset_index(drop=True)
    
    ## Check nearby Gaia source
    queryG = """select q3c_dist(x.ra,x.dec,{0},{1})*3600. as q3c_d 
                from gaia_dr2.gaia_source as x 
                where q3c_radial_query(x.ra,x.dec,{0},{1},{2}/3600.)""".format(ra,dec,cmradius)
    nearbydataG = pd.DataFrame(sqlutil.get(queryG, **wsdb_kwargs))
    NG=len(nearbydataG)-1
    NG+=1*(NG<0)
    
    ## Check nearby DECAPS sources
    queryD = """select q3c_dist(x.ra,x.dec,{0},{1})*3600. as q3c_d 
                from decaps_dr1.main as x 
                where q3c_radial_query(x.ra,x.dec,{0},{1},{2}/3600.)""".format(ra,dec,cmradius)
    nearbydataD = pd.DataFrame(sqlutil.get(queryD, **wsdb_kwargs))
    ND=len(nearbydataD)-1
    ND+=1*(ND<0)
    
    ## Check nearby DECAPS sources
    queryGN = """select q3c_dist(x.ra,x.dec,{0},{1})*3600. as q3c_d, x.* 
                from jason_sanders.galacticnucleus as x 
                where q3c_radial_query(x.ra,x.dec,{0},{1},{2}/3600.)""".format(ra,dec,cmradius)
    nearbydataGN = pd.DataFrame(sqlutil.get(queryGN, **wsdb_kwargs))
    print(nearbydataGN)
    NGN=len(nearbydataGN)-1
    NGN+=1*(NGN<0)
    
    print('There are',
          NGN,'nearby GALACTICNUCLEUS sources,',
          ND,'nearby DECAPS sources,',
          NG,'nearby Gaia source and',
          NB,'nearby reliable VIRAC sources.')
    
    ## If no nearby sources, return unadulterated LC
    if (NB<1)&(NG<1)&(ND<1):
        if with_plot:
            GPX = vvvG[fld].values
            if fld=='seeing':
                GPX=1./GPX
            dPC=np.linspace(np.min(GPX),np.max(GPX))
            plt.errorbar(GPX,vvvG['mag'],yerr=vvvG['error'],fmt='o')
            plt.legend()
            plt.xlim(0.1,)
            plt.xlabel(r'$\chi^2$')
            plt.xlabel(r'1/ (Seeing / arcsec)')
            plt.ylabel('Magnitude')
        neighbourinfo = [NB, NG, ND, 0, (NB>0)|(NG>0)|(ND>0)]
        return vvvG, ra, dec, None, None, None, neighbourinfo
    
    ## Find observations when neighbour detected
    vvv_neighbour = [load_virac_lc(s, extra_info=['seeing']) 
                     for s in nearbydata['sourceid'].values[1:]]
    # Nearby detections not in main lightcurve
    vvvG['flgX']=1
    extra_lc = [pd.merge(vvvG[['mjdobs','flgX']],vvv,on='mjdobs',how='right') 
                for vvv in vvv_neighbour]
    for i in range(len(extra_lc)):
        extra_lc[i]=extra_lc[i][extra_lc[i]['flgX']!=extra_lc[i]['flgX']].reset_index(drop=True)
        del extra_lc[i]['flgX']
    del vvvG['flgX']
    if len(extra_lc)>0:
        extra_lc = pd.concat(extra_lc).reset_index(drop=True)
    for ii in range(len(vvv_neighbour)):
        vvv_neighbour[ii]['flg']=1
    vvvJ = [pd.merge(vvvG,vvv[['mjdobs','flg']],on='mjdobs',how='left') for vvv in vvv_neighbour]

    counts = np.zeros_like(vvvG['mag'].values,dtype='int')
    ## 2**k sum means we count the combinations (1:1,2:2,3:1+2,4:3,5:1+3,6:2+3,7:1+2+3,etc.)
    for ivv,v in enumerate(vvvJ):
        counts+=(v['flg']==1)*2**ivv
    
    groups = [counts==c for c in np.unique(counts)]
    
    vvvG2 = vvvG.copy()
    offs = np.ones(len(groups))*np.nan
    lbls=['No neighbour']
    for gg in range(len(groups)-1):
        lbls+=['neighbour %i'%(gg+1)]
    
    ## Detrend two groups (with/without neighbour detected)
    
#     if np.count_nonzero(fltr)==0 and NB>0:
#         if with_plot:
#             plt.errorbar(vvv_neighbour[0]['seeing'],
#                          vvv_neighbour[0]['mag'],
#                          yerr=vvv_neighbour[0]['error'],
#                          fmt='o',color='k',zorder=10,label='neighbour lc')
    
    neighbourinfo = [NB, NG, ND, np.sum([1*(np.count_nonzero(v['flg']>0)>0) for v in vvvJ]), (NB>0)|(NG>0)|(ND>0)]
    
    for ii,(fltr,l) in enumerate(zip(groups,lbls)):
        if np.count_nonzero(fltr)<2:
            continue
        GPX = vvvG[fld].values[fltr]
        if fld=='seeing':
            GPX=1./GPX
        if log:
            GPX = np.log10(GPX)
        ## Sometimes will fail to do Cholesky (not sure why -- perhaps bounds on parameters need to be set)
        try:
            GPP=run_gp(GPX,vvvG['mag'].values[fltr],vvvG['error'].values[fltr],
                   kernel=george.kernels.ExpSquaredKernel(metric=5),
                      )
        except:
            GPP = george.GP(
                george.kernels.ExpSquaredKernel(metric=5),
                mean=np.median(vvvG['mag'].values[fltr]),
                white_noise=np.log(np.var(vvvG['mag'].values[fltr])))
            GPP.compute(vvvG['HJD'].values[fltr], vvvG['error'].values[fltr],)

        if with_plot:
            dPC=np.linspace(np.min(GPX),np.max(GPX))
            yy=GPP.predict(vvvG['mag'].values[fltr],dPC-np.median(GPX),return_cov=False)
            plt.errorbar(GPX,vvvG['mag'][fltr],yerr=vvvG['error'][fltr],fmt='o',label=l,
                        color=sns.color_palette(n_colors=20)[ii])
            plt.plot(dPC,yy,color=sns.color_palette(n_colors=20)[ii])

        vvvG2.loc[fltr,'mag']-=GPP.predict(vvvG['mag'].values[fltr],GPX-np.median(GPX),return_cov=False)
        mid = np.percentile(GPX,90.)
        if log:
            mid=np.log10(mid)
        offs[ii]=GPP.predict(vvvG['mag'].values[fltr],mid-np.median(GPX),return_cov=False)

    group_counts = [np.count_nonzero(counts) for counts in groups]
    for gg,oo in zip(group_counts,offs):
        off = oo
        if gg>4:
            break
    if ~np.isfinite(off):
        off = offs[0]
    vvvG2['mag']+=off
    
    if with_plot:
        plt.legend()
        plt.xlim(0.1,)
        plt.xlabel(r'$\chi^2$')
        plt.xlabel(r'1/ (Seeing / arcsec)')
        plt.ylabel('Magnitude')
        
    return vvvG2, ra, dec, nearbydata, vvv_neighbour, extra_lc, neighbourinfo


def load_corrected_virac_lc(sid, radec=None, with_plot=True):
    vvv = load_virac_lc(sid, extra_info=["seeing"])
    vvv = quality_cut_phot(vvv)
    if len(vvv) == 0:
        return vvv, vvv, None
    vvv_, ra, dec, nearbydata, nearby_lc, extra_lc, neighbourinfo = \
        correct_nearby(vvv, sid, radec=radec, cmradius=1., with_plot=with_plot)
    vvv_ = sigclipper(vvv_)
    vvv = sigclipper(vvv)
    return vvv_, vvv, nearbydata, neighbourinfo
