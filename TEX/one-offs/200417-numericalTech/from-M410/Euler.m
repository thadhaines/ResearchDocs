function [y,x] = Euler(f,a,ya,b,n)

x = zeros(1,n+1);
y = zeros(1,n+1);
x(1) = a;
y(1) = ya;
h = (b-a)/n;
for j=1:n
  y(j+1) = y(j) +h*f(x(j),y(j));
  x(j+1) = x(j) + h;
end

end

