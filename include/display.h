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

class Display {

    private:
        cairo_surface_t* screen;
        unsigned char* buffer;
        int width;
        int height;

    public:

        static Display* getInstance();
        static gboolean onDraw(GtkWidget* widget, cairo_t* cr, gpointer data);

        cairo_status_t init(int width, int height);
        void run();
        void destory();

        cairo_surface_t* getScreen() { return screen; }
        int getWidth() { return width; }
        int getHeight() { return height; }
        unsigned char* getBuffer() { return buffer; }

};


#endif
