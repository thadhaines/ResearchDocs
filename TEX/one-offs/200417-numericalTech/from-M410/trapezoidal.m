function [yInt] = trapezoidal(x,y)
    ts = x(2)-x(1);
    yInt = 0;
    
    for ndx=2:max(size(y))
        yInt = yInt+ (y(ndx)+y(ndx-1))/2*ts;
        
    end
    
end