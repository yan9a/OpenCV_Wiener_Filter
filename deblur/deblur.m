function deblur()
%Deblur a blurred and noisy image using Wiener filter
pkg load image %load image package if necessary
I=imread('blur.jpg');% read an image
%I=rgb2gray(I); % convert to grayscale

% show the original image in grayscale
figure;
imshow(I); 
title('Original');

I=im2double(I); % convert to double type
F=fft2(I); % discrete fourier transform
%show the Fourier spectrum
%figure;
%imshow(log(abs(fftshift(F))+1),[ ]); 
%title('Fourier spectrum');

[ih iw]=size(I);%get size
% To estimate SNR
% take low freq part as signal

H=Butterworth(ih,iw,100,2,0); %get Butterworth filter

G=F.*fftshift(H); %filter in freq domain
g=ifft2(G); % inverse discrete fourier transform
img=abs(g); % get the absolute value

% show the resulting low freq image
%figure;
%imshow(img); 
%title('Signal');  

% take high freq part as noise
in=I-img;
% show the noise image
%figure;
%imshow(in); 
%title('Noise');  

%get signal power
sp=sum(sum(img.*img));

%get noise power
np=sum(sum(in.*in));

%signal to noisesignal ratio
snr=sp/np;
snr

H=Gaussian(ih,iw,50,0);  %get blur spectrum which is Gaussian filter 
H=fftshift(H);
Fdb=WienerFilter(F,H,1/snr); %deblur using Wiener filter
gdb=ifft2(Fdb); % inverse discrete fourier transform

idb=im2uint8(abs(gdb)); % convert grayscale

% show the resulting deblurred image
figure;
imshow(idb); 
title('Deblurred image');  

%imwrite(idb,'deblur.jpg');% save the image  
end

function Fd=WienerFilter(G,H,NSR)
%Wiener_Filter performs linear image restoration 
%Wiener filter estimates f that minimize the statistical error function
%G= input image spectrum
%H= blurring spectrum
%NSR= noise to signal ratio
AH2=H.*conj(H);% get magnitude square
%F=((1./H).*(AH2./(AH2+NSR))).*G;
Fd=((AH2./(AH2+NSR))./H).*G;
%show the Fourier spectrum
%figure;
%imshow(log(abs(fftshift(Fd))+1),[ ]); 
%title('Deblurred spectrum');
end

function H=Gaussian(ih,iw,s,filterType)
    %Gaussian_Filter performs high-pass or low-pass Gaussian filtering
    % ih,iw= height and width, 
    % s= CutOff frequency (standard deviation), 
    % filterType= 0- Low-pass and 1- High-pass
    %produce u and v coordinate
    u1=(0:1:ih-1)-floor(ih/2);
    v1=(0:1:iw-1)-floor(iw/2);
    [V,U]=meshgrid(v1,u1);
    %produce filter for low pass
    H=e.^(-(U.^2+V.^2)./(2*(s^2)));    
    if (filterType==1) %if high-pass
        H=1-H;
    end    
end

function H=Butterworth(ih,iw,Do,n,filterType)
    %Butterworth_Filter performs high-pass or low-pass butterworth filtering
    % ih,iw= height and width,  
    % Do= CutOff frequency, 
    % n= order, 
    % filterType= 0- Low-pass and 1- High-pass    
    %produce u and v coordinate
    % u1=(-floor(ih/2):1:(ceil(ih/2)-1));
    % v1=(-floor(iw/2):1:(ceil(iw/2)-1));
    u1=(0:1:ih-1)-floor(ih/2);
    v1=(0:1:iw-1)-floor(iw/2);
    [V,U]=meshgrid(v1,u1);
    fra=(U.^2+V.^2)./(Do^2); % for low pass
    if (filterType==1) % if high pass
        fra=1./fra;
    end        
    H=1./(1+fra.^n); %get mask
end
