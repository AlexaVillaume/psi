
import sys
import os
import glob
import subprocess
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from combined_model import CombinedInterpolator
from spi.utils import dict_struct, within_bounds
from spi.plotting import get_stats, quality_map, bias_variance, specpages, write_results

from combined_params import bounds, features, pad_bounds


def cumulative_all(types):
    fname = 'spi_cumulative_rms_jacknife.pdf'

    regimes = {'Cool Dwarfs': ['#CC3333', '-'],
               'Cool Giants': ['#FF6600', '--'],
               'Warm Dwarfs': ['#006633','-.'],
               'Warm Giants': ['#66CC00', ':'],
               'Hot Stars':   ['#000099', '-']}

    fig = plt.figure(figsize=(12,4))
    axes = [plt.subplot(1,3,1), plt.subplot(1,3,2), plt.subplot(1,3,3)]

    for i, (ax, t) in enumerate(zip(axes, types)):
        for d in t:
            with h5py.File(d, 'r') as hf:

                ## I need to learn how to use regex
                tmp = d.split('/')
                tmp2 = tmp[len(tmp)-1]
                if i == 0:
                    regime = tmp2.replace('_unc=True_cwght=0.000_results.h5','')
                elif i == 1:
                    regime = tmp2.replace('_unc=True_cwght=0.010_results.h5','')
                elif i == 2:
                    regime = tmp2.replace('_unc=True_cwght=0.001_results.h5','')
                regime = regime.replace('_', ' ')

                wave = np.array(hf['wavelengths'])
                observed = np.array(hf['observed'])
                predicted = np.array(hf['predicted'])
                snr = np.array(hf['observed'])/np.array(hf['uncertainty'])
                feh = hf['parameters']['feh']
                labels = np.array(hf['parameters'])


            inbounds = within_bounds(bounds[regime], labels)

            wmin=0.4
            wmax=2.5
            telluric = [(1.32, 1.41), (1.82, 1.94)]
            varinds = (wave > wmin) & (wave < wmax)
            for bl, bh in telluric:
                varinds = varinds & ((wave  > bh) | (wave  < bl))

            delta = predicted[inbounds,:]/observed[inbounds,:] - 1.0
            var_total = np.nanvar(delta[:, varinds], axis=1)

            rms = np.sqrt(var_total)*100
            rms[~np.isfinite(rms)]=1000
            oo = np.argsort(rms)


            if regime == 'Hot Stars':
                ax.plot(rms[oo], np.arange(len(oo))/float(max(oo))*100,
                    lw=5, color=regimes[regime][0], ls=regimes[regime][1],
                    label=regime)
            else:
                ax.plot(rms[oo], np.arange(len(oo))/float(max(oo))*100,
                        lw=3, color=regimes[regime][0], ls=regimes[regime][1],
                        label=regime)


        if i == 0:
            ax.annotate('C3K weight 0.000', xy=(1,0),
                         xycoords='axes fraction', xytext=(0.95, 0.05),
                         ha='right', va='bottom', fontsize=12)
        if i == 1:
            ax.annotate('C3K weight 0.001', xy=(1,0),
                         xycoords='axes fraction', xytext=(0.95, 0.05),
                         ha='right', va='bottom', fontsize=12)
        if i == 2:
            ax.annotate('C3K weight 0.010', xy=(1,0),
                         xycoords='axes fraction', xytext=(0.95, 0.05),
                         ha='right', va='bottom', fontsize=12)

        ax.axvline(5, lw=1, ls='--',color='#424242')
        ax.axhline(90, lw=1, ls='--',color='#424242')

        ax.set_xlim(0,40)
        ax.set_xlabel('Fractional RMS (%)', fontsize=14)

    axes[0].set_ylabel('%(<RMS)', fontsize=14)
    axes[0].legend(frameon=False, loc='lower right', fontsize=12)
    plt.tight_layout()

    plt.suptitle('Jack knife tests')
    plt.savefig(fname)

    plt.show()


def check_chromo_emm():
    cd = 'v8_test/Hot_Stars_unc=True_cwght=0.010_results.h5'
    with h5py.File(cd, 'r') as hf:
        c3k = (np.array(hf['parameters']['miles_id']) == 'c3k')
        md  = (np.array(hf['parameters']['miles_id']) == 'mdwarf')
        lib = ((np.array(hf['parameters']['miles_id']) != 'mdwarf') &
               (np.array(hf['parameters']['miles_id']) != 'c3k'))

        wave = np.array(hf['wavelengths'])
        observed = np.array(hf['observed'])
        predicted = np.array(hf['predicted'])
        snr = np.array(hf['observed'])/np.array(hf['uncertainty'])
        names = np.array(hf['parameters']['name'])
        feh = np.array(hf['parameters']['feh'])


    #mlib = '../PSITables/culled_libvtest_w_conv_mdwarfs_w_unc_tc.h5'
    #with h5py.File(mlib, 'r') as hf:
    #    names = np.array(hf['ancillary']['name'])
    #    wave = np.array(hf['wavelengths'])
    #    observed = np.array(hf['spectra'])
    #    feh = np.array(hf['parameters']['feh'])
    hd = [0.404, 0.416]
    hg = [0.4284, 0.4420]
    hb = [0.4829, 0.4892]
    ha = [0.6550, 0.6650]
    i = ((wave > hd[0]) & (wave < hd[1]))
    k = ((wave > hg[0]) & (wave < hg[1]))
    l = ((wave > hb[0]) & (wave < hb[1]))
    m = ((wave > ha[0]) & (wave < ha[1]))
    with PdfPages('../SPS_Plots/PSICheck/check_chrom_emm_hs.pdf') as pdf:
        for j, name in enumerate(names):
            fig = plt.figure(figsize=(10,10))
            plt.suptitle('Star: {0}, Fe/H: {1}'.format(name, feh[j]))
            ax1 = plt.subplot(2,2,1)
            ax2 = plt.subplot(2,2,2)
            ax3 = plt.subplot(2,2,3)
            ax4 = plt.subplot(2,2,4)

            coeffs = np.polynomial.chebyshev.chebfit(wave[i], observed[j][i], 1)
            poly = np.polynomial.chebyshev.chebval(wave[i],
                                coeffs)
            ax1.plot(wave[i], observed[j][i]/poly, color='k', label='obs')

            coeffs = np.polynomial.chebyshev.chebfit(wave[i], predicted[j][i], 1)
            poly = np.polynomial.chebyshev.chebval(wave[i],
                                coeffs)
            ax1.plot(wave[i], predicted[j][i]/poly, color='g', label='psi')

            ax1.legend(loc='lower left', frameon=False)

            coeffs = np.polynomial.chebyshev.chebfit(wave[k], observed[j][k], 1)
            poly = np.polynomial.chebyshev.chebval(wave[k],
                                coeffs)
            ax2.plot(wave[k], observed[j][k]/poly, color='k')

            coeffs = np.polynomial.chebyshev.chebfit(wave[k], predicted[j][k], 1)
            poly = np.polynomial.chebyshev.chebval(wave[k],
                                coeffs)
            ax2.plot(wave[k], predicted[j][k]/poly, color='g')

            coeffs = np.polynomial.chebyshev.chebfit(wave[l], observed[j][l], 1)
            poly = np.polynomial.chebyshev.chebval(wave[l],
                                coeffs)
            ax3.plot(wave[l], observed[j][l]/poly, color='k')

            coeffs = np.polynomial.chebyshev.chebfit(wave[l], predicted[j][l], 1)
            poly = np.polynomial.chebyshev.chebval(wave[l],
                                coeffs)
            ax3.plot(wave[l], predicted[j][l]/poly, color='g')

            coeffs = np.polynomial.chebyshev.chebfit(wave[m], observed[j][m], 1)
            poly = np.polynomial.chebyshev.chebval(wave[m],
                                coeffs)
            ax4.plot(wave[m], observed[j][m]/poly, color='k')

            coeffs = np.polynomial.chebyshev.chebfit(wave[m], predicted[j][m], 1)
            poly = np.polynomial.chebyshev.chebval(wave[m],
                                coeffs)
            ax4.plot(wave[m], predicted[j][m]/poly, color='g')

            #ax1.axvline(0.410314, ls='--', color='k')
            #ax1.axvline(0.410189, ls='--', color='k')
            #ax1.axvline(0.43419, ls='--', color='k')
            #ax1.axvline(0.434203, ls='--', color='k')
            pdf.savefig()

def chromo_emm_v_teff():
    cd = '../PSITables/culled_libv5_w_mdwarfs_w_unc_w_allc3k.h5'

    with h5py.File(cd, 'r') as hf:
        type_ = (np.array(hf['parameters']['miles_id']))

        wave = np.array(hf['wavelengths'])
        observed = np.array(hf['spectra'])
        feh = np.array(hf['parameters']['feh'])
        logt = np.array(hf['parameters']['logt'])
        logg = np.array(hf['parameters']['logg'])
        logl = np.array(hf['parameters']['logl'])


    c3k = ((type_ == 'c3k') & (logt > np.log10(4000.)) &
           (logt < np.log10(6000.)) & (logg > 3.5))
    md = ((type_ == 'mdwarf') & (logt > np.log10(4000.)) &
            (logt < np.log10(6000.)) & (logg > 3.5))
    lib = ((type_ != 'c3k') & (type_ != 'mdwarf') &
            (logt < np.log10(6000.)) & (logt > np.log10(4000.)) & (logg > 3.5))

    fig = plt.figure(figsize=(12,4))
    # Hgamma
    ax1 = plt.subplot(1,3,1)
    i = ((wave > 0.432003) & (wave < 0.436378))
    lib_hg = [np.average(star[i]) for star in observed[lib]]
    c3k_hg = [np.average(star[i]) for star in observed[c3k]]
    md_hg = [np.average(star[i]) for star in observed[md]]

    ax1.plot(logt[lib], lib_hg/10**logl[lib], ls='None', marker='o', label='IRTF')
    ax1.plot(logt[md], md_hg/10**logl[md], ls='None', marker='o', label='Mann')
    ax1.plot(logt[c3k], c3k_hg/10**logl[c3k], ls='None', marker='o', label='C3K')

    ax1.set_ylabel('Flux @ Hg')

    plt.legend(loc='upper left')

    # Hdelta
    ax2 = plt.subplot(1,3,2)
    i = ((wave > 0.408376) & (wave < 0.412252))
    lib_hg = [np.average(star[i]) for star in observed[lib]]
    c3k_hg = [np.average(star[i]) for star in observed[c3k]]
    md_hg = [np.average(star[i]) for star in observed[md]]

    ax2.plot(logt[lib], lib_hg/10**logl[lib], ls='None', marker='o', label='IRTF')
    ax2.plot(logt[md], md_hg/10**logl[md], ls='None', marker='o', label='Mann')
    ax2.plot(logt[c3k], c3k_hg/10**logl[c3k], ls='None', marker='o', label='C3K')

    ax2.set_ylabel('Flux @ Hd')

    # Between
    ax3 = plt.subplot(1,3,3)
    i = ((wave > 0.418) & (wave < 0.421))
    lib_hg = [np.average(star[i]) for star in observed[lib]]
    c3k_hg = [np.average(star[i]) for star in observed[c3k]]
    md_hg = [np.average(star[i]) for star in observed[md]]

    ax3.plot(logt[lib], lib_hg/10**logl[lib], ls='None', marker='o', label='IRTF')
    ax3.plot(logt[md], md_hg/10**logl[md], ls='None', marker='o', label='Mann')
    ax3.plot(logt[c3k], c3k_hg/10**logl[c3k], ls='None', marker='o', label='C3K')

    ax3.set_ylabel('Flux between')

    plt.tight_layout()
    plt.savefig('flux_v_teff_wd.pdf')


def residual_look(real):
    regimes = {'Cool Dwarfs': '#CC3333', 'Cool Giants': '#FF6600',
               'Warm Dwarfs': '#006633', 'Warm Giants': '#66CC00',
               'Hot Stars': '#000099'}

    with PdfPages('spi_test.pdf') as pdf:
        fig = plt.figure(figsize=(15,10))
        for d in real:
            with h5py.File(d, 'r') as hf:
                tmp = d.split('/')
                tmp2 = tmp[len(tmp)-1]
                regime = tmp2.replace('_unc=True_cwght=0.010_results.h5','')
                regime = regime.replace('_', ' ')
                wave = np.array(hf['wavelengths'])
                observed = np.array(hf['observed'])
                predicted = np.array(hf['predicted'])
                snr = np.array(hf['observed'])/np.array(hf['uncertainty'])
                feh = hf['parameters']['feh']
                labels = np.array(hf['parameters'])

                inbounds = within_bounds(bounds[regime], labels)

                wmin=0.75
                wmax=2.5
                telluric = [(1.32, 1.41), (1.82, 1.94)]
                varinds = (wave > wmin) & (wave < wmax)
                for bl, bh in telluric:
                    varinds = varinds & ((wave  > bh) | (wave  < bl))

                delta = predicted[inbounds,:]/observed[inbounds,:] - 1.0
                from astropy.stats import median_absolute_deviation
                unc = median_absolute_deviation(delta, axis=0)

                plt.suptitle(regime, fontsize=20)

                ax1 = plt.subplot(2,3,1)
                reg = [0.35, 0.5]
                i = ((wave > reg[0]) & (wave < reg[1]))
                ax1.plot(wave[i], unc[i])
                ax1.set_ylim(0.0, 0.2)

                ax2 = plt.subplot(2,3,2)
                reg = [0.5, 0.8]
                i = ((wave > reg[0]) & (wave < reg[1]))
                ax2.plot(wave[i], unc[i])
                ax2.set_ylim(0, 0.2)

                ax3 = plt.subplot(2,3,3)
                reg = [0.8, 1.3]
                cat1 = 0.8500
                cat2 = 0.8544
                cat3 = 0.8664
                mg = 0.8808
                feh = 0.9920
                ki1 = 1.1692
                ki2 = 1.1779
                i = ((wave > reg[0]) & (wave < reg[1]))
                ax3.plot(wave[i], unc[i])
                ax3.axvline(cat1, ls='--', color='#424242')
                ax3.axvline(cat2, ls='--', color='#424242')
                ax3.axvline(cat3, ls='--', color='#424242')
                ax3.axvline(mg, ls='--', color='#424242')
                ax3.axvline(feh, ls='--', color='#424242')
                ax3.axvline(ki1, ls='--', color='#424242')
                ax3.axvline(ki2, ls='--', color='#424242')
                ax3.set_ylim(0, 0.08)

                ax4 = plt.subplot(2,3,4)
                reg = [1.45, 1.75]
                i = ((wave > reg[0]) & (wave < reg[1]))
                ax4.plot(wave[i], unc[i])
                ax4.set_ylim(0, 0.08)

                ax5 = plt.subplot(2,3,5)
                reg = [1.95, 2.1]
                cai1 = 1.9782
                cai2 = 1.9862
                i = ((wave > reg[0]) & (wave < reg[1]))
                ax5.plot(wave[i], unc[i])
                ax5.axvline(cai1, ls='--', color='#424242')
                ax5.axvline(cai2, ls='--', color='#424242')
                ax5.set_ylim(0, 0.08)

                ax6 = plt.subplot(2,3,6)
                reg = [2.1, 2.4]
                co1 = 2.296
                co2 = 2.3245
                i = ((wave > reg[0]) & (wave < reg[1]))
                ax6.plot(wave[i], unc[i])
                ax6.axvline(co1, ls='--', color='#424242')
                ax6.axvline(co2, ls='--', color='#424242')
                ax6.set_ylim(0, 0.08)

                plt.tight_layout()
                pdf.savefig()
                plt.cla()
                plt.clf()


if __name__=='__main__':
    #check_chromo_emm()
    #sys.exit()

    #residual_look(real)
    #sys.exit()


    eirtfv2_c3k0_000 = glob.glob('EIRTFv2_results/*0.000*h5')
    eirtfv2_c3k0_001 = glob.glob('EIRTFv2_results/*0.001*h5')
    eirtfv2_c3k0_010 = glob.glob('EIRTFv2_results/*0.010*h5')

    cumulative_all([eirtfv2_c3k0_000, eirtfv2_c3k0_001, eirtfv2_c3k0_010])

