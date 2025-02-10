package flip

import (
	"fmt"

	"github.com/checkioname/cnn-skin-model/internal/augmentation"
	"gocv.io/x/gocv"
)

// Flip flips a 2D array around horizontal(0), vertical(1), or both axes(-1).
type Flip struct {
	flipCode int
}

func NewFlip() augmentation.Augmentor {
	return &Flip{
		flipCode: -1,
	}
}

func (f *Flip) GetName() string {
	return "flipped"
}

func (f *Flip) AugmentImage(img gocv.Mat) gocv.Mat {
	fmt.Println("Flipping img")

	flippedImg := gocv.Mat{}
	gocv.Flip(img, &flippedImg, f.flipCode)

	return flippedImg
}
