# Binary Finder
## Classification tool for identifying binary stars from PSF models

An exquisite Point Spread Function (PSF) determination for the [Euclid](http://www.euclid-ec.org/) mission is crutial. One of the factor hampering the reconstruction is the presence of multiple stars in the star field. We propose a tool to identify those binaries based on the repeated measurements of the light profile.

## Abstract of the upcoming paper

_Context_. Measuring weak gravitation lensing signal to the level required by next generation of space-based surveys demands exquisite point spread function (PSF) reconstruction. However, the presence of unresolved binary stars can significantly distort the PSF shape.

_Aims_. In an effort to mitigate this significant bias, we aim at detecting unresolved binaries in realistic Euclid stellar populations. We tests out methods in numerical experiments where (i) the PSF shape is known to Euclid requirements and (ii) the PSF shape is unknown.

_Methods_. We draw simulated observations of PSF shapes for this proof-of-concept paper. As per Euclid survey plan, the objects are observed four times. We propose three methods to detect unresolved binary stars. The detection is based on the systematic and correlated biases between exposures of the same object. One method is a simple correlation analysis while the two others (random forest and artificial neural network) use supervised machine learning.

_Results_. In both experiments, we demonstrate the capacity of our methods to detect unresolved binary stars on simulated measurements. The performance depends on the level of a priori knowledge of the PSF shape. Good detection performances are observed in both experiences. Full complexity of the images and the survey design are not included, but key aspects of a more mature pipeline are discussed.

_Conclusions_. Finding unresolved binaries in the objects used for PSF reconstruction increases the quality of the PSF determination at arbitrary positions. We show, using different approach, the capacity of detecting at least the most damaging binary stars for the PSF reconstruction process.


Key words. Methods: data analysis – Methods: statistical – (Stars:) binaries (including multiple): close

## Running the code

The scripts that were used to generate the results of the paper are in the `script/` repository. The simulated PSFs are internal to the [Euclid Consoritum](http://www.euclid-ec.org/).

## Other codes from EPFL / LASTRO

To discover other codes from the astrophysics laboratory of [EPFL](http://www.epfl.ch), you can go [there](http://lastro.epfl.ch/software)
