function [y,x] = ralston(f,a,ya,b,n)
  x = zeros(1,n+1);
  y = zeros(1,n+1);
  x(1) = a;
  y(1) = ya;
  h = (b-a)/n;
  for i=1:n
    k1 = f(x(i),y(i));
    k2 = f(x(i)+2*h/3, y(i)+2*h/3*k1);
    y(i+1) = y(i) + h/4*(k1+3*k2);
    x(i+1) = a + (b-a)*i/n;
  end
end

