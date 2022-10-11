Xline= [0,  10];
Yline= [0,  2];
Zline= [2, 1];

color = 'blue';
p = plot3(Xline,Yline,Zline,color);
p.LineWidth = 3;
hold on;
%p1= plot3(Xline,Yline,[0,0],color);
p2= plot3(Xline,[0,0],Zline,color);
p3= plot3([0,0],Yline,Zline,color);
%p1.LineStyle = '--';
p2.LineStyle = '--';
p3.LineStyle = '--';
%p1.Color(4)  = 0.5;
p2.Color(4)  = 0.5;
p3.Color(4)  = 0.5;
%p1.LineWidth = 2;
p2.LineWidth = 2;
p3.LineWidth = 2;
hold off;

alpha0=0.6;


name = 'isocontour_eta=0.0_eps=0.3_q=3.0_kappa=2.0_delta=0.7_dR0dr=0.0_skappa=0.0_sdelta=0.0.hdf5';

% import stuff
omn_mat     = h5read(name, '/omnv' );
alpha_mat   = h5read(name, '/alphav' );
s_mat       = h5read(name, '/sv' );
AE_mat      = h5read(name, '/AEv' );

AE_max = max(AE_mat,[],'all') ;
AE_min = 0.0 ;

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

daspect([1,1,1]);
xlabel('$\hat{\omega}_n$','Interpreter','latex');
ylabel('$\alpha$','Interpreter','latex');
zlabel('$s$','Interpreter','latex');
grid on;

view([137 20]);




colormap(turbo);
c = colorbar;
caxis([AE_min AE_max]);
c.Label.String = '$\widehat{A}$';
c.Label.Interpreter = 'latex';

txt = '$(a)$: $\delta = +0.7$';
text(0.5,1.5,-1.5,txt,'Interpreter','latex')


fontname(gcf,"CMU Serif")

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 6, 2.1]);
exportgraphics(gcf,'LH-plot-postriangle.png','Resolution',2000)