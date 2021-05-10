#ifndef DISPLAY_H
#define DISPLAY_H

#include <gtkmm/drawingarea.h>
#include <gtkmm/application.h>
#include <gtkmm/window.h>
#include <cairomm/context.h>
/*
#include <cairo.h>
#include <gtk/gtk.h>
*/

/*
 * Display class to render talk with the OS and 
 * draw a imagebuffer in a window
 */
class Display {

    private:
        cairo_surface_t* screen;
        unsigned char* buffer;
        GtkWidget *darea;
        int width;
        int height;

    public:

        // Singleton method to get the display instance
        static Display* getInstance();
        // Callback method to handle OS draw requests
        static gboolean onDraw(GtkWidget* widget, cairo_t* cr, gpointer data);

        // Initializes the display
        cairo_status_t init(int width, int height);

        // Runs the display
        void run();

        // Cleans up the display
        void destory();

        // Gets a pointer to the cairo object for the screen
        cairo_surface_t* getScreen() { return screen; }

        void redraw();

        // Gets the width in pixels of the screen
        int getWidth() { return width; }

        // Gets the height in pixels of the screen
        int getHeight() { return height; }

        // Renders to a png
        void writeToPNG(const char* filename);

        // Gets a pointer to the underlying frame buffer, in BGRA format
        unsigned char* getBuffer() { return buffer; }

};


#endif
