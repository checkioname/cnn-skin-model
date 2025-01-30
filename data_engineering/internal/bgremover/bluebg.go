package bgremover

import (
	"os"
	"strings"

	"gocv.io/x/gocv"
)

// Como o fundo de nossos arquivos sao azuis, vamos remover somente fundos azuis :)


func RemoveBlueBgDir(dirPath string) {
  entries, _ := os.ReadDir(dirPath) 

  for _, entry := range entries {
    filename := strings.Split(entry.Name(), "/")
    // outpath := strings.Join(filename[:len(filename) -1], "/no_blue_bg")
    outpath := "no_blue_bg" + filename[len(filename) -1]

    result := RemoveBlueBg(entry.Name()) 
    gocv.IMWrite(outpath, result)
  }
}



func RemoveBlueBg(imgPath string) gocv.Mat {

  img := gocv.IMRead(imgPath, gocv.IMReadColor)
	if img.Empty() {
		panic("Erro ao carregar a imagem")
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
	defer result.Close()
	gocv.BitwiseAndWithMask(img, img, &result, maskInv)

	// Salvar a imagem final sem fundo azul
	// outputPath := "output_no_blue.png"
	// gocv.IMWrite(outputPath, result)

	return result
}
