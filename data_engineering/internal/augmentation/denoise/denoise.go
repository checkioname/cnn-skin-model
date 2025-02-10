package denoise

import (
	"fmt"

	"github.com/checkioname/cnn-skin-model/internal/augmentation"
	"gocv.io/x/gocv"
)

type Denoiser struct {
	h                  float32
	hColor             float32
	templateWindowSize int
	searchWindowSize   int
}

func NewDenoise() augmentation.Augmentor {
	return &Denoiser{
		h:                  10.0,
		hColor:             10.0,
		templateWindowSize: 5,
		searchWindowSize:   19,
	}
}

func (d *Denoiser) GetName() string {
	return "denoised"
}

func (d *Denoiser) AugmentImage(img gocv.Mat) gocv.Mat {
	fmt.Println("Denoising img")
	denoised := gocv.NewMat()
	gocv.FastNlMeansDenoisingColoredWithParams(img, &denoised, d.h, d.hColor, d.templateWindowSize, d.searchWindowSize)

	gray := gocv.NewMat()
	gocv.CvtColor(denoised, &gray, gocv.ColorBGRToGray)

	equalized := gocv.NewMat()
	gocv.EqualizeHist(gray, &equalized)

	equalizedRGB := gocv.NewMat()
	gocv.CvtColor(equalized, &equalizedRGB, gocv.ColorGrayToBGR)

	denoised.Close()
	gray.Close()
	equalized.Close()

	return equalizedRGB
}
