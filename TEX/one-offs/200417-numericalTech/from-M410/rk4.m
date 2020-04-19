function [y,x] = rk4(f,a,ya,b,n)
  x = zeros(1,n+1);
  y = zeros(1,n+1);
  x(1) = a;
  y(1) = ya;
  h = (b-a)/n;
  for i=1:n
    k1 = f(x(i), y(i));
    k2 = f(x(i)+h/2, y(i)+h/2*k1);
    k3 = f(x(i)+h/2, y(i)+h/2*k2);
    k4 = f(x(i)+h, y(i)+h*k3);
    y(i+1) = y(i) + h/6*(k1+2*k2+2*k3+k4);
    x(i+1) = a + (b-a)*i/n;
  end
end