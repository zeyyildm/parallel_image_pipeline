//adım 1: görseli okuma
#include <opencv2/opencv.hpp>
#include <vector> //matrise çevirmek için

cv::Mat toGrayFloat01(const cv::Mat& bgr){ //& kopyalamıyoruz orijinal görüntüde çalışıyoruz
    CV_Assert(bgr.type() == CV_8UC3); //yanlışlıkla farklı tip gelirse diye
    //her kanal 8 bit C3 = 3 kanal(RGB)
    //yani bu satır ile gerçekten renkli bi görüntü geldi mi ona bakıyorum
    cv::Mat gray(bgr.rows, bgr.cols, CV_32F); //gri görüntü tek kanal ve her piksel 32-bit float
    for(int r = 0; r < bgr.rows; r++){ //görüntüyü satır satır gez
        const cv::Vec3b* srcRow = bgr.ptr<cv::Vec3b>(r); //ilk pikselin adresini alıyoruz
        float* dstRow = gray.ptr<float>(r); //Artık dstRow[c] dediğimde r satırı, c.sütunundaki gri değer demek

        for(int c = 0; c < bgr.cols; c++){ //iç döngü ile sütunda geziyoruz
            float B = srcRow[c][0]; //rgb değerleri char gelir
            float G = srcRow[c][1];
            float R = srcRow[c][2];

            float y = (0.114f * B + 0.587f * G + 0.299f * R) / 255.0f; //biz griye dönüştürdük
            //255 ile 0-1 arasına getirdik
            dstRow[c] = y; //gri değeri gray görüntüsüne gidip attım

        }
    }
    return gray; //gri görüntü döndü
}


//PROCESS (matris çarpımı)
cv::Mat applyConvolution(const cv::Mat& input) { //matris çarpımını sağlar
    cv::Mat output = cv::Mat::zeros(input.size(), input.type()); // Çıktı matrisi oluştur (Girişle aynı boyutta, float tipinde)

    float kernel[3][3] = { //kenar bulma matrisi 
        {-1.0f, -1.0f, -1.0f},
        {-1.0f,  8.0f, -1.0f},
        {-1.0f, -1.0f, -1.0f}
    };

    //dört tane for döngüsü iç içe olduğu için işlemciyi en çok yoran ve paralleştirmeye en uygun kısımdır
    for (int r = 1; r < input.rows - 1; r++) { //iç içe dört for -> satır, sütun, kernel satır, kernel sütun
        for (int c = 1; c < input.cols - 1; c++) {
            float sum = 0.0f;
            
            // Kernel Matrisini (3x3) Resmin o anki parçasıyla çarpıyoruz her piksel için ayrı ayrı işlem yapılır
            for (int kr = -1; kr <= 1; kr++) { //kernel satırı
                for (int kc = -1; kc <= 1; kc++) { //kernel sütunu
                    float pixelVal = input.at<float>(r + kr, c + kc); //resimden o anki komşu piksel alınır
                    float kernelVal = kernel[kr + 1][kc + 1]; //kernaldei karşılığı alınır
                    sum += pixelVal * kernelVal; //çarpıp toplanır
                }
            }
            //mutlak değer alınır sayı 0 ile 1 arasında sınırlandırılır
            output.at<float>(r, c) = std::min(std::max(std::abs(sum), 0.0f), 1.0f); 
            
        }
    }
    return output;
}

//POSTPROCESS
//görüntü siyah-beyaz (binary) hale getirilir
cv::Mat applyThreshold(const cv::Mat& input, float thresholdValue) { 
    cv::Mat output = input.clone(); //resmin kopyasını oluşturduk orijinal veriyi bozmamak için
    
    for (int r = 0; r < output.rows; r++) {
        float* rowPtr = output.ptr<float>(r); //her seferinde adrese gitmek yerine o satırın başlangıç adresini alıyoruz
        for (int c = 0; c < output.cols; c++) {
            //eğer piksel değeri eşikten büyükse 1 (beyaz) küçükse 0 (siyah) yap
            if (rowPtr[c] > thresholdValue) {
                rowPtr[c] = 1.0f;
            } else {
                rowPtr[c] = 0.0f;
            }
        }
    }
    return output;
}


int main(){
    cv::Mat img = cv::imread("araba.jpg", cv::IMREAD_COLOR); //görseli oku
    if(img.empty()){ //görsel okunabildi mi
        //burda okunan görsel RGB, 8 BİT, 3 KANALLI
        std::cerr << "ımage yuklenemedııııı" << "\n";
        return (1);
    }

    double start_time = (double)cv::getTickCount(); //zaman ölçümü başlangıcı

    //PREPROCESS
    cv::Mat gray01 = toGrayFloat01(img);
    //burdaki gray01 TEK KANALLI DEĞERLER -1 ARASI 32 BİT
    //ama bu şekilde görseli kaydedemeyiz jpg 8 bit ister
    //kontrol için tekrar 8 bite dönüştürüp kaydedeceğiz ki kontrol sağlayalım griye dönmüş mü diye
    cv::Mat gray8u;
    gray01.convertTo(gray8u, CV_8U, 255.0);
    cv::imwrite("serial_gray_copy.png", gray8u); //görseli düzgün okkuyabildik mi diye bir kopyasını kaydediyorum

    //PROCESS
    cv::Mat edges = applyConvolution(gray01);

    //POSTPROCESS
    cv::Mat finalResult = applyThreshold(edges, 0.30f);

    //zaman ölçümü sonu
    double end_time = (double)cv::getTickCount();
    double time_needed = (end_time - start_time) / cv::getTickFrequency();
    std::cout << "seri islem okey" << "\n";
    std::cout << "Gecen Sure: " << time_needed * 1000 << " ms" << "\n"; //saniyeyi milisaniyeye çevirdik
    
    //sonucu kaydetme
    cv::Mat saveImg;
    finalResult.convertTo(saveImg, CV_8U, 255.0);
    
    cv::imwrite("sonuc_serial.png", saveImg); 
    std::cout << "Sonuc gorseli 'sonuc_serial.png' olarak kaydedildi." << "\n";

    return 0;
}