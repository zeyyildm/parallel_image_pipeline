//her process kendi belleğine sahip
//görüntüler processler arasında paylaştırılıyor

//master -> rank 0: Görüntüleri toplayıp okuyacak işleri dağıtacak
// worker -> rank 1,2,3,...: Görüntü işleme görevlerini yapacak

//thread değil process mantığı kullanacağız
//1-) MPI_Bcast
//2-) MPI_Scatter: İş paylaşımı dağıtımı yapacak
//3-) MPI_Gather: İşlem sonuçlarını toplayacak
//4-)MPI_Finalize: en uzun processten süreyi ölçeceğiz

//adım 1: görseli okuma
#include <mpi.h>
#include <iostream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <opencv2/opencv.hpp>

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
//kernel = küçük bir matris
//bu fonksiyon kernelı her piksele uygular.::::
//ARABANIN SINIRLARI, ÇİZGİLERİ, DETAYLARI ORTAYA ÇIKARIR
cv::Mat applyConvolution(const cv::Mat& input) { //matris çarpımını sağlar
    cv::Mat output = cv::Mat::zeros(input.size(), input.type()); // Çıktı matrisi oluştur (Girişle aynı boyutta, float tipinde)

    float kernel[3][3] = { //kenar bulma matrisi 
        {-1.0f, -1.0f, -1.0f}, //merkez piksel 8 komşular -1
        {-1.0f,  8.0f, -1.0f}, //yoğun değişim olan yerler = kenar
        {-1.0f, -1.0f, -1.0f}
    };
    //r, c -> hangi pikseldeyiz
    //kr, kc o pikselin etrafındaki komşuların hangisine bakıyoruz
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
            //abs -> negatif kenarları pozitife çevirir
            //max -> negatif olmasın
            //min -> 1'i geçmesin
            
        }
    }
    return output; //tek kanallı, float, kenarları vurgulanmış
}
//matris çarpımı görüntüden anlamlı bilgiyi süzer

//POSTPROCESS
//görüntü siyah-beyaz (binary) hale getirilir
cv::Mat applyThreshold(const cv::Mat& input, float thresholdValue) { 
    //bu fonksiyon ile o piksel kenar sayılacak kadar güçlü mü bunun kararını veriyoruz
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


//gelen her rank aynı maini çalıştırır
//hepsi ayrı process ayrı bellek alanı kullanır
int main(int argc, char** argv){
    MPI_Init(&argc, &argv); //mpi başlatılır
    int rank = 0; //ben kimim
    int size = 1; //kaç process var

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //her processin kendi rankini alır
    MPI_Comm_size(MPI_COMM_WORLD, &size); //toplam process sayısını alır.

    //diğer processlerin inputPaths değişkeniyle bağlantısı olmaz birbirlerinden bağımsızdır
    int N = 0;    //başlangıçtaki image sayısı

    float thresholdValue = 0.45f;
    if(rank == 0){ //sadece rank = 0 görüntü listesini okuyabilir. çünkü herkes aynı işi yapmamalı
        if(argc <= 1){
            std::cerr << "Kullanim: mpirun -np P ./pipeline_mpi img1 img2 ...\n";
        }
        else{
            N = argc - 1; //görüntü sayısı
        }
        //şu an rank 1-2-3 N= 0 görür
    }

    MPI_Bcast(&thresholdValue, 1, MPI_FLOAT, 0, MPI_COMM_WORLD); //bir processteki veriyi alır ve tüm processlere kopyalar
    //eşik değeri tüm processlere yolladık
    //eşik değeri bir pikselin önemli mi önemsiz mi olduğuna karar vermemize yarıyordu
    //kenar mı değil mi eşik değeri sayesinde karar veriyorduk
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD); //aynı mantıkla N i gönderdik
    if(N == 0){ //yani hiç image verilmemiş
        MPI_Finalize(); //programı düzgünce kaptır
        return 0;
    }

    //SCATTER İLE İNDEXLERİ DAĞITMA
    //görevi işi eşit parçalar halinde processlere dağıtmak
    //scatter tüm processler tarafından çağrılmak zorundadır

    int chunk = (N + size - 1) / size; //her processin alacağı iş miktarı
    //N=5 size=3 (rank 0-1-2) chunk =  = 2
    //her process en fazla 2 görüntü işleyecek
    std::vector<int> localIdx(chunk, -1); //her processte olacak.  bu değişken o processe düşen indexleri tutacak
    std::vector<int> sendIdx; // sadece rank0. tüüm image indexlerini sırayla tutar

    if(rank == 0){
        sendIdx.assign(size * chunk, -1); //[-1, -1, -1, -1, -1, -1]
        for(int i = 0; i < N; i++){
            sendIdx[i] = i; //[0, 1, 2, 3, 4, -1] -1 ise orda iş yok demek
            //böyle bir şey  yapmamızın sebebi scatter her processten aynı sırada ver bekler biz de fazla yerlere -1 koyarız
        }
    }

    MPI_Scatter(rank==0 ? sendIdx.data() : nullptr,
                chunk, MPI_INT,
                localIdx.data(),
                chunk, MPI_INT,
                0, MPI_COMM_WORLD);

    double t0 = MPI_Wtime(); //zaman ölçümünü baslatir
    int local_count = 0; //her processin kaç tane image işlediğini sayacak

    for (int k = 0; k < chunk; k++) { //chunkların yani işlerin içinde gezen döngü
        int idx = localIdx[k]; //bu processin işleyeceği image indexi
        if (idx < 0) 
            continue; //burda iş yok devam et

        // argv[1 + idx] = o image’ın path’i
        std::string path = argv[1 + idx]; //terminalden gelen dosya yolunu alir ./mpi ilk argüman olacağı için +1

        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR); //görentüyü oku
        if (img.empty()) { //yoksa, bozuksa, okunamıyorsa hata mesajı at
            std::cerr << "rank " << rank << "] okunamadi: " << path << "\n";
            continue;
        }

        cv::Mat gray01 = toGrayFloat01(img);
        cv::Mat edges  = applyConvolution(gray01);
        cv::Mat finalR = applyThreshold(edges, thresholdValue);

        cv::Mat saveImg;
        finalR.convertTo(saveImg, CV_8U, 255.0);
        //seride yaptığımız işlemlerin aynısını yapıyoruz

        // En basit output ismi: idx ve rank ile çakışmaz
        std::string outName = "mpi_r" + std::to_string(rank) +
                              "_i" + std::to_string(idx) + "_result.png"; //processler aynı dosya ismini yazmasın diye hazır kod aldık. amaç karışıklık çıkmasın
        cv::imwrite(outName, saveImg); //görüntüye diske yaz

        local_count++; //maine giren process 1 image işledi
    }

    double local_time = MPI_Wtime() - t0; //her processin kendi lokal süresi

    // 4) Total time = max(local_time) (Reduce)
    double total_time = 0.0;

    //REDUCE NE YAPAR?
    //tüm processlerin local_time değişkenşini al ve en büyüğünü seç çünkü gerçek süre en uzun süren process kadardır. bu en büyük sonucu da rank 0'a koy
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "MPI total time (MAX rank): " << total_time*1000.0 << " ms\n";
    }


    MPI_Finalize();    
    return (0);
}

//threadler de processler de aslında işi bölüşür. ama threadler tek bir processin içindeki alt görevlerdir ve aynı belleği paylaşır
//processler ise birbirinden tamamen bağımsızdır ve kendi bellek alanlarına sahiptirler
//threadlerle veri paylaşımı otomatiktir.
//threadlerde imagei herkes görürken processlerde ise kimse görmüyordu ben kendim dağıttım
//processler arası iletişim MPI kütüphanesi ile sağlanır
//MPI_Bcast: bir processteki veriyi alır ve tüm processlere kopyalar
//MPI_Scatter: bir processte olan veriyi parçalara böler ve her processse bir parça gönderir
//MPI_Gather: her processten bir parça alır ve tek bir processte toplar
//MPI_Reduce: her processten bir değer alır ve tek bir processte toplar (toplama, çarpma, maksimum, minimum gibi işlemler yapılabilir)
//MPI_Finalize: MPI ortamını sonlandırır