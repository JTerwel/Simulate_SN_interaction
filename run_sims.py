'''
A program to run different simsurvey simulations quickly

Author: Jacco Terwel
Date 09-08-2022

- First version
- Should note down paper references in a readme once this goes to GitHub
- Should also link the locations of the used files
- Setting the strength, start & duration of the interaction now goes via a csv file
'''

#Imports
#Simsurvey imports
import simsurvey
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15
import astropy.units as u
import sncosmo
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
#Model making specific imports
from scipy import interpolate
#Imports needed to save values properly
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
#Import to track elapsed time
from time import time as tracktime


def main():
	tnow = tracktime()
	#Set locations of files to load in
	master_loc = Path('/home/jaccoterwel/Documents/simsurvey')
	sfd98_dir = master_loc / 'sfd98'#!!!
	#Load the 11fe model
	sn_part1 = load_TimeSeriesSource(master_loc / '11fe_raw_cubic_peak.csv',
									 '11fe_cubic')
	sn_part2 = load_TimeSeriesSource(master_loc / '11fe_raw_lin_late_interp.csv',
									 '11fe_lin_with_interp_specs',
									 time_spline_degree=1)
	#Create a flat background to ensure simulated non-detections
	flat_backg = sncosmo.TimeSeriesSource(np.linspace(-500, 2000, 4),
										  sn_part1._wave,
										  np.ones((4, len(sn_part1._wave)))*1e-15,
										  name='background')
	#Set fixed parts for the feature
	wl0 = 6563 #Angstrom H alpha line
	sigma = 30 #Angstrom width
	#Set host extinction effects for the model
	effects = [sncosmo.CCM89Dust()]
	effect_names = ['host']
	effect_frames = ['rest']
	#Load the survey plan, keep fields for later use
	fields, plan = load_survey(master_loc, 'year1-3_public_plan_ZTF.csv',
							   'ZTF_Fields.txt', 'ZTF_quadrant_corners.txt')
	#Set parameters for the generator
	zrange = (0.0, 0.1)
	ra_range = (0, 360) #degrees
	dec_range = (-30, 90) #degrees
	mjd_range = (58195, 58487) #18-03-2018 -- 04-01-2019
	ntransient = 100000 #Nr. of transients to simulate
	n_det = 3 #Nr. of detections needed to find an object
	tresh = 5 #S/N treshold for a detection
	#Set observatory location to Palomar (used for saving the lcs)
	observatory = EarthLocation(lat=33.35627096604836*u.deg,
								lon=-116.86481294596469*u.deg, height=1700*u.m)
	#List all simulations & run them parallel to each other
	interact_params = pd.read_csv(master_loc / 'interaction_parameters.csv', header=0)
	arglist = []
	for _ in range(len(interact_params)):
		arglist += [[sn_part1, sn_part2, flat_backg, interact_params.strength.loc[_],
					 interact_params.start.loc[_], interact_params.duration.loc[_],
					 wl0, sigma, effects, effect_names, effect_frames, ntransient,
					 ra_range, dec_range, mjd_range, zrange, sfd98_dir, plan, n_det,
					 tresh, master_loc, fields, observatory]]
	print(f'Loading completed in {tracktime()-tnow} seconds\nStarting the simulations loop')
	pool = mp.Pool(mp.cpu_count())
	list(tqdm(pool.imap_unordered(run_sim, arglist), total=len(arglist)))
	pool.close()
	pool.join()
	return

#*--------------------------------*
#| Functions for simulation setup |
#*--------------------------------*

def register_band(old_name, new_name, min_trans=0):
	'''
	Register a new band by modifying an existing one
	
	old_name (string): Name of existing band to use as a base
	new_name (string): Name of the new band
	min_trans (float): transmission value to clip the band at, default = 0
	'''
	old = sncosmo.get_bandpass(old_name)
	new = sncosmo.Bandpass(old.wave[old.trans>min_trans],
						   old.trans[old.trans>min_trans], name=new_name)
	sncosmo.register(new)
	return

def load_TimeSeriesSource(loc, name, time_spline_degree=3):
	'''
	Load a TimeSeriesSource & update its spline degree in the time direction
	Note: From sncosmo version 2.7.0 this can be done sncosmo function itself

	loc (Path): Location of the source file
	name (string): Name of the TimeSeriesSource
	time_spline_degree (int): Spline degree in the time direction, default = 3
	'''
	pd_source = pd.read_csv(loc, header=0, index_col=0)
	wl = pd_source.columns.values.astype('float64')
	source = sncosmo.TimeSeriesSource(pd_source.index.values, wl,
									  pd_source.values, name=name)
	if time_spline_degree != 3:
		source._model_flux = interpolate.RectBivariateSpline(pd_source.index.values,
															 wl, pd_source.values,
															 kx=time_spline_degree)
	return source

def load_survey(loc, obsfile, fieldsfile, ccdfile):
	'''
	Load the survey data & create the survey plan

	loc (Path): Locations of the files to load
	obsfile (string): Observation plan file
	fieldsfile (string): Observation fields file
	ccdfile (string): ccd corners file
	'''
	obs = pd.read_csv(loc / obsfile, header=0)
	#Change to use correct filters (makes sure ztfg & ztfi can be found more easily)
	obs.loc[obs['band'] == 'ztfg', 'band'] = 'ZTF_g'
	obs.loc[obs['band'] == 'ztfr', 'band'] = 'ZTF_r'
	obs.loc[obs['band'] == 'ztfi', 'band'] = 'ZTF_i'
	fields = pd.read_fwf(loc / fieldsfile, header=0)
	ccd_corners = np.genfromtxt(loc / ccdfile, skip_header=1)
	#Each set of 4 lines contains the 4 corners of a ccd
	ccds = [ccd_corners[4*k:4*k+4, :2] for k in range(64)]
	#Make the survey plan
	plan = simsurvey.SurveyPlan(time=obs.time.values-2400000.5,
								band=obs.band.values,
								obs_field=obs.field.values,
								obs_ccd=obs.ccd.values,
								skynoise=obs.skynoise.values,
								comments=obs.comment.values,
								fields={'ra': fields.RA.values,
										'dec': fields.Dec.values,
										'field_id': fields.ID.values},
								ccds=ccds)
	return fields, plan

def gauss(x, wl0, a, sigma):
	#A simple Gaussian function
	return a*np.exp(-(x-wl0)**2 / (2*sigma**2))

def create_feature(wls, phases, wl0, a, sigma, name):
	'''
	Create a Gaussian feature TimeSeriesSource

	wls (array): Wavelengths of the TimeSeriesSource
	phases (array): Phases of the TimeSeriesSource
	wl0 (float): Rest wavelength of Gaussian peak
	a (array): Amplitude of feature at each phase
	sigma (float): Width of the Gaussian peak
	name (string): Feature name
	'''
	feature = gauss(wls, wl0, 1, sigma)
	flux_mat = np.zeros((len(phases), len(wls)))
	for i in range(len(phases)):
		flux_mat[i] = a[i]*feature
	return sncosmo.TimeSeriesSource(phases, wls, flux_mat, name=name)

#*---------------------------------------*
#| Functions for use during a simulation |
#*---------------------------------------*

def run_sim(args):
	#Run a simulation
	#Register bandpasses under the name that is going to be used (has to be done here due to multiprocessing)
	register_band('ztfg', 'ZTF_g', 0.01)
	register_band('ztfr', 'ZTF_r')
	register_band('ztfi', 'ZTF_i', 0.01)
	#Unpack args
	sn_part1, sn_part2, flat_backg, strength, start, duration, wl0, sigma, effects, effect_names, effect_frames, ntransient, ra_range, dec_range, mjd_range, zrange, sfd98_dir, plan, n_det, tresh, master_loc, fields, observatory = args
	#create the feature
	phases = np.linspace(start, start+duration, 5)
	amps = np.ones_like(phases) * 0.003 * strength
	feat_name = f'Flat_{strength}15cp_{start}_{duration}_Ha'
	feature = create_feature(sn_part1._wave, phases, wl0, amps, sigma, feat_name)
	#Combine source parts
	comp_source = simsurvey.CompoundSource((sn_part1, sn_part2, flat_backg, feature),
										   name='11fe_with_feature')
	#Make the model
	model = sncosmo.Model(comp_source, effects=effects, effect_names=effect_names,
						  effect_frames=effect_frames)
	#Set the transient properties & create the generator & survey
	transientprop = {'lcmodel':model, 'lcsimul_func':random_params}
	tr = simsurvey.get_transient_generator(zrange, ntransient=ntransient,
										   ra_range=ra_range, dec_range=dec_range,
										   mjd_range=mjd_range,
										   transientprop=transientprop,
										   apply_mwebv=True, ratefunc=rate,
										   sfd98_dir=sfd98_dir)
	survey = simsurvey.SimulSurvey(generator=tr, plan=plan, n_det=n_det,
								   threshold=tresh)
	#Adjust the filter gain
	for _ in survey.instruments:
		survey.add_instrument(_, gain=6.5)
	#Generate the light curves
	lcs = survey.get_lightcurves()
	#Filter the obtained light curves
	#lcs = lcs.filter(filterfunc) #Not used for now
	#Save the light curves in FPbot-style DataFrames
	saveloc = master_loc / f'sims/str{strength}start{start}dur{duration}'#!!!
	saveloc.mkdir(exist_ok=True)
	save_lcs(lcs, saveloc, fields, observatory)
	return

def random_params(redshifts, model, r_v=2., ebv_rate=0.11, **kwargs):
	#Change the flux amplitude based on redshift
	amp = 10**(-0.4*(Planck15.distmod(redshifts).value))
	param_dict = {}
	for s in model.param_names:
		if 'amplitude' in s:
			param_dict[s] = amp
	param_dict['hostr_v'] = r_v * np.ones(len(redshifts))#Cardelli, Clayton, Mathis 1989 (same as simsurvey paper)
	param_dict['hostebv'] = np.random.exponential(ebv_rate, len(redshifts))#Stanishev et al. 2018 (same as simsurvey paper)
	return param_dict

def rate(z):
	#Volumetric rate function
	return 2.4e-5 # 1/Mpc**3/yr, Frohmaier 2019

#*------------------------------------------------*
#| Functions for handling the output light curves |
#*------------------------------------------------*

def filterfunc(lc):
	#Filter function to be used in lcs.filter()
	#NOT USED FOR NOW!!!
	return lc

def flux2mag(flux, fluxerr, zp):
	#Convert flux, fluxerr to mag, magerr, upper limmit
	#Give 5 sigma upper limit when flux < 5*fluxerr, value = 99: not applicable
	mag = np.ones_like(flux)*99
	magerr = np.ones_like(flux)*99
	uplim = np.ones_like(flux)*99
	mask = flux>=5*fluxerr
	mag[mask] = -2.5*np.log10(flux[mask])+zp[mask]
	magerr[mask] = np.abs(-2.5*fluxerr[mask] / (flux[mask]*np.log(10)))
	uplim[~mask] = -2.5*np.log10(5*fluxerr[~mask])+zp[~mask]
	return mag, magerr, uplim

def calc_airmass(ra, dec, t, observatory):
	#Calculate the rmass at a given location & time
	pointings = SkyCoord(ra, dec, unit='deg')
	return [pointings[i].transform_to(AltAz(obstime=t[i], location=observatory)).secz for i in range(len(t))]

def save_fpbot_df(lc, savename, fields, observatory):
	#Save the lightcurve in an FPbot-style csv file
	df = pd.DataFrame()
	#Calculate derived values
	mags, mag_errs, uplims = flux2mag(np.array(lc['flux']), np.array(lc['fluxerr']),
									  np.array(lc['zp']))
	t = Time(lc['time'], format='mjd')
	airmass = calc_airmass([fields[fields.ID==i].RA.values[0] for i in lc['field']],
						   [fields[fields.ID==i].Dec.values[0] for i in lc['field']],
						   t, observatory)
	decday = [f'{i.isot[0:4]}{i.isot[5:7]}{i.isot[8:10]}{str(float(i.isot[11:13])/24+float(i.isot[14:16])/(24*60)+(float(i.isot[17:]))/(24*3600))[2:8]}' for i in t]
	#Put all values in the DataFrame & save it
	df['obsmjd'] = lc['time']
	df['filter'] = lc['band']
	df['Fratio'] = lc['flux'] / 10**(0.4*lc['zp'])
	df['Fratio.err'] = lc['fluxerr'] / 10**(0.4*lc['zp'])
	df['mag'] = mags
	df['mag_err'] = mag_errs
	df['upper_limit'] = uplims
	df['ampl.err'] = lc['fluxerr']
	df['chi2dof'] = 1
	df['seeing'] = 2
	df['magzp'] = [25.8 if i[-1]=='g' else 25.9 if i[-1]=='r' else 25.5 for i in lc['band']]
	df['magzprms'] = 0.04
	df['airmass'] = airmass
	df['nmatches'] = [130 if i>=19 else 20 for i in -2.5*np.log10(5*lc['fluxerr'])+lc['zp']]
	df['rcid'] = lc['ccd']
	df['fieldid'] = lc['field']
	df['infobits'] = 0
	df['filename'] = [f"ztf_{decday[i]}_{'%06.f'%lc['field'][i]}_z{lc['band'][i][-1]}_c{'%02.f'%int(lc['ccd'][i]/4+1)}_o.fits" for i in range(len(decday))]
	df.to_csv(savename)
	return

def save_lcs(lcs, saveloc, fields, observatory):
	#Save the light curves in a pickle file & FPbot-style DataFrames
	lcs.save(saveloc / 'lcs.pkl')
	for _ in range(len(lcs.lcs)):
		save_fpbot_df(lcs[_], saveloc / f'lc_{_}.csv', fields, observatory)
	return

if (__name__ == '__main__'):
	main()
