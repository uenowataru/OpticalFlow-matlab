%Img1 = [0:5; 2:7; 0:5; 2:7; 2:7]';
%Img2 = [-2:3; 0:5; -2:3; 0:5; 0:5]';

%Make image gradients
[ x, y ] = meshgrid(1:20, 1:20);
x = x - 10;
y = y - 10;
Img1 = x.*x + y.*y;
Img1 = Img1 ./ max(max(Img1));
x = x - 1;
%y = y - 1;
Img2 = x.*x + y.*y;
Img2 = Img2 ./ max(max(Img2));

numrows = size(Img1,1);
numcols = size(Img1,2);


%%%%%%%%%%%%%%
% Make an outlier
Img2(3,2) = Img2(3,2) + 0.1;

Wi = [1,0;0,1];
%should be size (imgx-1 * imgy-1, 2)
N = numel(Img1);
A = zeros(2*N,2);
bvector = zeros(2*N,1);

%plot([Img1 Img2])
Ev = 0.01^2;
Ef = 0.1^2;

dImg1x = [ diff(Img1, 1, 2) zeros(numrows, 1) ];
dImg1y = [ diff(Img1, 1, 1); zeros(1, numcols) ];

B = (Img1 - Img2);

% Initial inlier/outlier status
zv = ones(N, 1);

pgaussian = @(x, Lambda) sqrt(det(1/(2*pi)*Lambda)) * exp(-0.5*x'*Lambda*x);

for i = 0:9
    %for total number of pixels
    for numpix = 0: (N-1)
        %add vector to the next empty row of A
        Ai = [ dImg1x(mod(numpix,numrows)+1, floor(numpix/numrows)+1) ...
            dImg1y(mod(numpix,numrows)+1, floor(numpix/numrows)+1) ];
        weightV = sqrt(zv(numpix+1));
        weightF = sqrt((1-zv(numpix+1)));
        A(2*numpix+1, :) = weightV * Ai * Wi;
        bvector(2*numpix+1) = weightV * B(mod(numpix,numrows)+1, floor(numpix/numrows)+1);
        A(2*numpix+2, :) = weightF * Ai * Wi;
        bvector(2*numpix+2) = weightF * B(mod(numpix,numrows)+1, floor(numpix/numrows)+1);
    end
    
    [ Qa, Ra ] = qr( [ A bvector ] );
    
    R = Ra(:, 1:end-1);
    d = Ra(:, end);
    
    v = R\d
    
    % Update z
    %for total number of pixels
    for numpix = 0: (N-1)
        grad = [ dImg1x(mod(numpix,numrows)+1, floor(numpix/numrows)+1) ...
            dImg1y(mod(numpix,numrows)+1, floor(numpix/numrows)+1) ];
        e = -B(mod(numpix,numrows)+1, floor(numpix/numrows)+1);
        
        x = e + grad* Wi*v;
        pv = pgaussian(x, 1/Ev) * 0.8;
        pf = pgaussian(x, 1/Ef) * 0.2;
        if pv + pf > 1e-15
            zv(numpix+1) = pv / (pv + pf);
        else
            zv(numpix+1) = 0;
        end
    end
    
    ZV = reshape(zv, size(Img1))
end

imshow(ZV);

