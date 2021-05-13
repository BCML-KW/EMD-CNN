path = 'F:\Sleep_data\public_data\sleep-edfx_data\sleep-edfx-mat\';
savepath = 'F:\Sleep_data\public_data\sleep-edfx_data\sleep-edfx-mat\SC_EMD\new\';
X = dir([path 'SC*.mat']);
A=[];
for i = 1:size(X,1)
    name = X(i).name;
    i
    load([path X(i).name])
    for j = 1:length(hyp)
        
        sig = zscore(xx(3000*(j-1)+1:3000*j));
%         noise = wgn(3000,1,mean(abs(sig)));
%         noise2 = wgn(3000,1,0.5*mean(abs(sig)));
%         signoi = sig+noise;
%         signoi2 = sig+noise2;
%         input = [sig signoi signoi2];
%         imf = memd(input);
%         imf1 = squeeze(imf(1,:,:));
%         imf2 = squeeze(imf(2,:,:));
%         imf3 = squeeze(imf(3,:,:));
        [imf0, residual] = emd(sig, 'MaxNumIMF', 5);
        imf = [imf0 residual];
%         imf0 = real(bemd(complex(sig, noise)));
%         imf = [imf0(1,:); imf0(2,:); imf0(3,:); imf0(4,:); imf0(5,:); imf0(6,:); imf0(7,:); imf0(8,:); imf0(9,:); sum(imf0(10:end,:), 1)];
        Label = hyp(j);
%         A{i,j} = size(imf,1);
        save([savepath name(1:end-14) '_' num2str(i, '%03i') '_' num2str(j, '%04i') '.mat'], 'sig', 'imf', 'Label');
%         for k = 1:6
%             subplot(3,2,k)
%             plot(imf(k,:))
%         end
    end
end
