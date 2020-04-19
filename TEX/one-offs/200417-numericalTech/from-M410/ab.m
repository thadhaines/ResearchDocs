function [y,x] = ab(f,a,ya,b,n)

x = zeros(1,n+1);
y = zeros(1,n+1);
x(1) = a;
y(1) = ya;
h = (b-a)/n;
for j=1:n
    if j <2
    y(j+1) = y(j) +h*f(x(j),y(j)); % do euler 
    %y(j+1) = y(1); % hold
    elseif j>2
    y(j+1) = y(j) +1.5*h*f(x(j),y(j))- 0.5*h*f(x(j-1),y(j-1));
        end
    x(j+1) = x(j) + h;
end

end

