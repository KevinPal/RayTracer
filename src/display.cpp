
#include "display.h"

#include <gtk/gtk.h>

/*
cairo_surface_t* Display::screen;
unsigned char* Display::buffer;
*/


gboolean Display::onDraw(GtkWidget* widget, cairo_t* cr, gpointer data) {
    Display* display = Display::getInstance();
    //printf("Drawing!\n");
    cairo_set_source_surface(cr, display->getScreen(), 0, 0);
    cairo_paint(cr);

    return FALSE;
}

// cairo_surface_write_to_png ()

Display* Display::getInstance() {
    static Display instance;
    return &instance;
}

cairo_status_t Display::init(int width, int height) {

    GtkWidget *window;
    GtkWidget *darea;
    cairo_status_t err;

    this->width = width;
    this->height = height;

    this->buffer = (unsigned char*) malloc(width * height * 4);
    memset((void*) buffer, 255, width * height * 4);
    this->screen = cairo_image_surface_create_for_data(buffer, CAIRO_FORMAT_ARGB32, width, height, width * 4);
    //screen = cairo_image_surface_create_from_png("yeet.png");

    if(err = cairo_surface_status(screen)) {
        return err;
    }

    //gtk_init(&argc, &argv);
    gtk_init(NULL, NULL);

    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);

    darea = gtk_drawing_area_new();
    gtk_container_add(GTK_CONTAINER (window), darea);

    g_signal_connect(G_OBJECT(darea), "draw", 
      G_CALLBACK(&Display::onDraw), NULL); 
    g_signal_connect(window, "destroy",
      G_CALLBACK (gtk_main_quit), NULL);

    gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);
    gtk_window_set_default_size(GTK_WINDOW(window), width, height); 
    gtk_window_set_title(GTK_WINDOW(window), "Image");

    gtk_widget_show_all(window);
}

void Display::run() {
    printf("run");
    gtk_main();
}

void Display::destory() {
    printf("destory");
    cairo_surface_destroy(screen);
    free(buffer);
}
