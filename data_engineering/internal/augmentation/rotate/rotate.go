package rotate

import (
	"fmt"
	"image"

	"github.com/checkioname/cnn-skin-model/internal/augmentation"
	"gocv.io/x/gocv"
)

type Rotator struct {
	rotationLevel int
}

func NewRotate() augmentation.Augmentor {
	return &Rotator{
		rotationLevel: 45,
	}
}

func (r *Rotator) GetName() string {
	return "rotaded"
}

func (r *Rotator) AugmentImage(img gocv.Mat) gocv.Mat {
	fmt.Println("Rotating img")

	center := image.Point{
		X: img.Cols() / 2,
		Y: img.Rows() / 2,
	}

	matrix := gocv.GetRotationMatrix2D(center, float64(r.rotationLevel), 1.0)

	rotated := gocv.NewMat()
	gocv.WarpAffine(img, &rotated, matrix, image.Point{X: img.Cols(), Y: img.Rows()})

	return rotated
}
