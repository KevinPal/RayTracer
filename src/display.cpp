
#include "display.h"

#include <gtk/gtk.h>


// Callback for an OS draw request, just draws the underlying frame buffer
// to the screen
gboolean Display::onDraw(GtkWidget* widget, cairo_t* cr, gpointer data) {
    Display* display = Display::getInstance();
    //printf("Drawing!\n");
    cairo_set_source_surface(cr, display->getScreen(), 0, 0);
    cairo_paint(cr);

    return FALSE;
}

// cairo_surface_write_to_png ()

// Singleton getInstance
Display* Display::getInstance() {
    static Display instance;
    return &instance;
}

// Sets up the display with a given with and height, and setsup
// the onDraw callback, and allocates the required bufferes
cairo_status_t Display::init(int width, int height) {

    GtkWidget *window;
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

void Display::redraw() {
    gtk_widget_queue_draw(darea);
}

void Display::writeToPNG(const char* filename) {
    cairo_surface_flush (screen);
    cairo_surface_write_to_png (screen, filename);
}

// Starts whatever is needed to actually display the screen
void Display::run() {
    gtk_main();
}

// Cleans up the display
void Display::destory() {
    printf("destory");
    cairo_surface_destroy(screen);
    free(buffer);
}
