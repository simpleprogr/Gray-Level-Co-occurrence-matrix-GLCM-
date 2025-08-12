function glcm_texture_analysis
    % Melatih model saat aplikasi dijalankan
    train_model();

    % Inisialisasi GUI
    fig = uifigure('Name', 'GLCM Texture Analysis', 'Position', [100 100 1000 700]);

    % Panel untuk identitas peneliti
    identityPanel = uipanel(fig, 'Title', 'PENELITI', 'Position', [10 380 403 310], 'FontSize', 12, 'TitlePosition', 'centertop');
    uilabel(identityPanel, 'Text', 'IMPLEMENTASI METODE GRAY LEVEL CO-OCCURRENCE MATRIX', 'Position', [10 248 383 30], 'FontSize', 12, 'HorizontalAlignment', 'center', 'FontName', 'Helvetica');
    uilabel(identityPanel, 'Text', 'MENGANALISA TEKSTUR KULIT WAJAH', 'Position', [10 224 383 30], 'FontSize', 12, 'HorizontalAlignment', 'center', 'FontName', 'Helvetica');
    uilabel(identityPanel, 'Text', 'RIZKY FAUZAN NAIBAHO', 'Position', [10 176 383 30], 'FontSize', 12, 'HorizontalAlignment', 'center', 'FontName', 'Helvetica');
    uilabel(identityPanel, 'Text', 'NPM : 009020019', 'Position', [10 152 383 30], 'FontSize', 12, 'HorizontalAlignment', 'center', 'FontName', 'Helvetica');
    uilabel(identityPanel, 'Text', 'PROGRAM STUDI TEKNOLOGI INFORMASI', 'Position', [10 104 383 30], 'FontSize', 12, 'HorizontalAlignment', 'center', 'FontName', 'Helvetica');
    uilabel(identityPanel, 'Text', 'FAKULTAS ILMU KOMPUTER DAN TEKNOLOGI INFORMASI', 'Position', [10 80 383 30], 'FontSize', 12, 'HorizontalAlignment', 'center', 'FontName', 'Helvetica');
    uilabel(identityPanel, 'Text', 'UNIVERSITAS MUHAMMADIYAH SUMATERA UTARA', 'Position', [10 68 383 30], 'FontSize', 12, 'HorizontalAlignment', 'center', 'FontName', 'Helvetica');
    uilabel(identityPanel, 'Text', 'MEDAN - 2024', 'Position', [10 44 383 30], 'FontSize', 12, 'HorizontalAlignment', 'center', 'FontName', 'Helvetica');

    % Panel untuk kontrol
    controlPanel = uipanel(fig, 'Title', 'PROSES', 'Position', [10 273 403 98], 'FontSize', 12, 'TitlePosition', 'centertop');
    loadBtn = uibutton(controlPanel, 'Text', 'PILIH CITRA', 'FontName', 'Helvetica', 'Position', [10 45 383 23], ...
                       'ButtonPushedFcn', @(loadBtn, event) load_image());
    calcBtn = uibutton(controlPanel, 'Text', 'PROSES GLCM', 'FontName', 'Helvetica', 'Position', [10 12 383 23], ...
                       'ButtonPushedFcn', @(calcBtn, event) calculate_glcm());

    % Panel untuk nilai GLCM
    resultPanel = uipanel(fig, 'Title', 'NILAI GLCM', 'Position', [10 52 403 211], 'FontSize', 12, 'TitlePosition', 'centertop');
    nilaiGLCMLabel = uilabel(resultPanel, 'Text', '', 'Position', [10 12 383 167], 'FontSize', 12, 'HorizontalAlignment', 'center', 'FontName', 'Helvetica');

    % Panel untuk gambar asli
    ax1 = uiaxes(fig, 'Position', [521 368 392 252]);
    title(ax1, 'CITRA ASLI');

    % Panel untuk gambar grayscale
    ax2 = uiaxes(fig, 'Position', [521 32 392 252]);
    title(ax2, 'CITRA GRAYSCALE');

    % Fungsi untuk melatih model
    function train_model()
        dataDir = fullfile(pwd, 'train');
        imageFolders = dir(dataDir);
        imageFolders = imageFolders([imageFolders.isdir]);
        imageFolders = imageFolders(~ismember({imageFolders.name}, {'.', '..'}));

        data_training = [];
        kelas = {};

        for i = 1:length(imageFolders)
            folderPath = fullfile(dataDir, imageFolders(i).name);
            imageFiles = dir(fullfile(folderPath, '*.jpg'));

            for j = 1:length(imageFiles)
                imgPath = fullfile(folderPath, imageFiles(j).name);
                img = imread(imgPath);
                
                % Periksa apakah gambar sudah grayscale
                if size(img, 3) == 3
                    grayImg = rgb2gray(img);
                else
                    grayImg = img;
                end
                
                glcm = graycomatrix(grayImg);
                stats = graycoprops(glcm);
                featureVector = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];

                data_training = [data_training; featureVector];
                kelas = [kelas; imageFolders(i).name];
            end
        end

        model = fitcknn(data_training, kelas, 'NumNeighbors', 1, 'Standardize', 1);
        save('trainedModel.mat', 'model');
    end

    % Fungsi untuk memuat gambar
    function load_image()
        [file, path] = uigetfile({'*.jpg;*.jpeg;*.png', 'Image Files (*.jpg, *.jpeg, *.png)'});
        if isequal(file, 0)
            return;
        end
        img = imread(fullfile(path, file));
        imshow(img, 'Parent', ax1);

        % Periksa apakah gambar sudah grayscale
        if size(img, 3) == 3
            grayImg = rgb2gray(img);
        else
            grayImg = img;
        end

        imshow(grayImg, 'Parent', ax2);
    end

    % Fungsi untuk menghitung dan menampilkan nilai GLCM serta klasifikasi
    function calculate_glcm()
        % Dapatkan gambar dari axes1
        img = getimage(ax1);

        if isempty(img)
            uialert(fig, 'Silakan muat gambar terlebih dahulu!', 'Error', 'Icon', 'error');
            return;
        end

        % Periksa apakah gambar sudah grayscale
        if size(img, 3) == 3
            grayImg = rgb2gray(img);
        else
            grayImg = img;
        end

        glcm = graycomatrix(grayImg);
        stats = graycoprops(glcm);
        featureVector = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];

        loadedData = load('trainedModel.mat');
        model = loadedData.model;
        predictedClass = predict(model, featureVector);

        nilaiGLCMLabel.Text = sprintf('Contrast: %.2f\nCorrelation: %.2f\nEnergy: %.2f\nHomogeneity: %.2f', ...
            stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity);

        uialert(fig, ['Kondisi kulit wajah terdeteksi sebagai : kulit ', predictedClass{1}], 'Hasil Klasifikasi', 'Icon', 'info');
    end
end
