Xline= [0,  10]
Yline= [0,  1]
Zline= [2, 0]
p=plot3(Xline,Yline,Zline,'black')
p.LineWidth = 3;
alpha0=0.6;

surf0=isosurface(omn_mat,alpha_mat,s_mat,AE_mat,AE_max/25);
p0 = patch(surf0);
isonormals(omn_mat,alpha_mat,s_mat,AE_mat,p0);
set(p0,'FaceColor','[0.2322    0.1855    0.4989]','EdgeColor','none','FaceAlpha',alpha0); % set the color


surf1=isosurface(omn_mat,alpha_mat,s_mat,AE_mat,AE_max/5);
p1 = patch(surf1);
isonormals(omn_mat,alpha_mat,s_mat,AE_mat,p1);
set(p1,'FaceColor','[0.2453    0.6073    0.9973]','EdgeColor','none','FaceAlpha',alpha0); % set the color


surf2=isosurface(omn_mat,alpha_mat,s_mat,AE_mat,2*AE_max/5);
p2 = patch(surf2);
isonormals(omn_mat,alpha_mat,s_mat,AE_mat,p2);
set(p2,'FaceColor','[0.2738    0.9704    0.5186]','EdgeColor','none','FaceAlpha',alpha0); % set the color


surf3=isosurface(omn_mat,alpha_mat,s_mat,AE_mat,3*AE_max/5);
p3 = patch(surf3);
isonormals(omn_mat,alpha_mat,s_mat,AE_mat,p3);
set(p3,'FaceColor','[0.8825    0.8663    0.2170]','EdgeColor','none','FaceAlpha',alpha0); % set the color


surf4=isosurface(omn_mat,alpha_mat,s_mat,AE_mat,4*AE_max/5);
p4 = patch(surf4);
isonormals(omn_mat,alpha_mat,s_mat,AE_mat,p4);
set(p4,'FaceColor','[0.9411    0.3562    0.0705]','EdgeColor','none','FaceAlpha',alpha0); % set the color

daspect([1,1,1])
xlabel('$\hat{\omega}_n$','Interpreter','latex')
ylabel('$\alpha$','Interpreter','latex')
zlabel('$s$','Interpreter','latex')
grid on

view([-43 12])




colormap(turbo);
c = colorbar;
caxis([AE_min AE_max]);
c.Label.String = '$\widehat{A}$';
c.Label.Interpreter = 'latex';


set(gcf, 'Units', 'Inches', 'Position', [0, 0, 6, 3.5])