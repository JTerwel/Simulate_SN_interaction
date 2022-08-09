# Simulate_SN_interaction
The code that is used to run simulations to test the Late-time_binning_program

This code is based on simsurvey. The code, relevant documentation, and relevand citations can be found [here](https://github.com/ZwickyTransientFacility/simsurvey).

The model used can be found [here](https://github.com/JTerwel/SN2011fe_model). A flat background is added between phases -500 - 2000 days to ensure observations when the SN is not (yet) active.

To run the code properly, the sfd98 dust maps were used to model extinction effects. The fields are defined using the ZTF fields, found here, and the individual ccd output channels using the quadrant corners. All of these can be found [here](https://github.com/ZwickyTransientFacility/simsurvey-examples). The observing plan is made using the year 1-3 observing plan of ZTF.

Other references:
- Volumetric rate: [Frohmaier et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.2308F/abstract)
- rv: [Cardelli, Clayton & Mathis, 1989](https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract)
- host E(B-V): [Stanishev et al. 2018](https://www.aanda.org/articles/aa/full_html/2018/07/aa32357-17/aa32357-17.html)
