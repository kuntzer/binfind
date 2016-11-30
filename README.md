# Binary Finder
## Classification tool for identifying binary stars from PSF models

An exquisite Point Spread Function (PSF) determination for the [Euclid](http://www.euclid-ec.org/) mission is crutial. One of the factor hampering the reconstruction is the presence of multiple stars in the star field. We propose a tool to identify those binaries based on the repeated measurements of the light profile.

## Abstract of the upcoming paper

_Context_. Exquisite point spread function (PSF) reconstruction is demanded to measure weak gravitation lensing signal to the level required by next generation of space-based surveys. Unresolved multiple stars can distort the true PSF shape by up to a few percent.

_Aims_. In an effort to mitigate this significant bias, we aim at detecting the unresolved binaries in realistic Euclid stellar populations in experiments where (i) the PSF shape is known to Euclid requirements and (ii) the PSF shape is unknown. We draw simulated observations of PSF shapes for this proof-of-concept paper. As per Euclid observation plan, the objects are observed four times.

_Methods_. We propose three methods to detect unresolved multiple stars. The detection is based upon the systematic and correlated biases between exposures of the same object. One method is a simple correlation analysis while the two others (random forest and artificial neural network) use supervised machine learning.

_Results_. In both experiments, we demonstrate the capacity of our approach to detect unresolved on simulated measurement at the Euclid level. If the PSF shapes are known, the metrics show better performance. Full complexity of the images and the survey design are not included, but key aspects of a more mature pipeline are discussed.

_Conclusions_. Finding unresolved binaries in the objects used for PSF reconstruction increases the quality of the PSF determination at arbitrary positions. We show, using different approach, the capacity of detecting at least the most harmful binaries.


Key words. Methods: data analysis – Methods: statistical – (Stars:) binaries (including multiple): close

## Running the code

The scripts that were used to generate the results of the paper are in the `script/` repository. The simulated PSFs are internal to the [Euclid Consoritum](http://www.euclid-ec.org/).

## Other codes from EPFL / LASTRO

To discover other codes from the astrophysics laboratory of [EPFL](http://www.epfl.ch), you can go [there](http://lastro.epfl.ch/software)
