package bgremover

import (
	"errors"
	"fmt"

	"gocv.io/x/gocv"
)

// Como o fundo de nossos arquivos sao azuis, vamos remover somente fundos azuis :)


type BlueBgRemover struct {}


func (br *BlueBgRemover) RemoveBG(imgPath string) (gocv.Mat, error) {
  fmt.Println(imgPath)
	img := gocv.IMRead(imgPath, gocv.IMReadColor)
	if img.Empty() {
		return gocv.Mat{}, errors.New("Erro ao carregar a imagem")

	}
	defer img.Close()

	// Converter a imagem para HSV
	hsvImg := gocv.NewMat()
	defer hsvImg.Close()
	gocv.CvtColor(img, &hsvImg, gocv.ColorBGRToHSV)

	// Definir limites para a cor azul no espaço HSV
	lowerBlue := gocv.NewScalar(100, 50, 50, 0)
	upperBlue := gocv.NewScalar(130, 255, 255, 0)

	// Criar uma máscara para a cor azul
	mask := gocv.NewMat()
	defer mask.Close()
	gocv.InRangeWithScalar(hsvImg, lowerBlue, upperBlue, &mask)

	// Inverter a máscara para preservar o objeto principal
	maskInv := gocv.NewMat()
	defer maskInv.Close()
	gocv.BitwiseNot(mask, &maskInv)

	// Aplicar a máscara invertida à imagem original
	result := gocv.NewMat()
	gocv.BitwiseAndWithMask(img, img, &result, maskInv)

	outputPath := "output_no_blue.png"
	gocv.IMWrite(outputPath, result)


	return result, nil
}
