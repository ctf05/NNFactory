function greyPlot = Interferogram_Plot(D, C, B, G, F, J, E, I, pixels, ~)

    % Number of pixels on each axis (square) {make sure you use an even number
    % of pixels}
    respix = pixels;
    
    % Defocus, Tilt(x), Tilt(y), Spherical, Coma(y), Coma(x), Astig(y), Astig(x), Piston
    % D         C          B        G          F        J       E          I        A
    
    
    FWF = ones(respix,respix);
    WFE = zeros(respix,respix);
    
    for i = 1:size(FWF,1)+1
        for j = 1:size(FWF,2)+1
            x = (i-(respix/2 + 1))/(respix/2);
            y = (j-(respix/2 + 1))/(respix/2);
    
            OPD = B*x + C*y + D*(x^2 + y^2) + E*(x^2 + 3*y^2) + F*y*(x^2 + y^2) + G*(x^2 + y^2)^2 + J*x*(x^2 + y^2) + I*(3*x^2 + y^2);
    
            WFE(i,j) = OPD;
        end
    end
    
    
    % imagesc(WFE,'Parent',OPDAxes)
    % colorbar(OPDAxes)
    % ccm = [ones(256,1),linspace(1,0,256)',linspace(1,0,256)'];
    % colormap(OPDAxes, ccm)
    % title(OPDAxes, "OPD (waves)")
    % xlim(OPDAxes,[0,respix])
    % ylim(OPDAxes,[0,respix])


    phase = 1 - abs(0.5 - (WFE - floor(WFE)))/0.5;
    % Convert directly to uint8 grayscale
    greyPlot = uint8(phase * 255);
end



% colorbar(PhaseAxes)
% colormap(PhaseAxes, colormap)
% title(PhaseAxes, "Phase Delta (waves)")
% xlim(PhaseAxes,[0,respix])
% ylim(PhaseAxes,[0,respix])
