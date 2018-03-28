#include "LPD8806.h"

int nLeds = 64;
int dataPin = 2;
int clockPin = 3;


LPD8806 strip = LPD8806(nLeds, dataPin, clockPin);

void setup() {
  strip.begin();
  strip.show();

}

void loop() {
colorChase(strip.Color(120,0,0),100,0);
}

void colorChase(uint32_t c, uint16_t wait, uint16_t b) {
  // Start by turning all pixels off:
  int i;
  //for(i=0; i<strip.numPixels(); i++) strip.setPixelColor(i, 0);
  lightRec(b,c,wait);
  //lightRec(b,strip.Color(120,50,50),wait);
 // lightRec(b,strip.Color(120,120,0),wait);
  //lightRec(b,strip.Color(20,20,120),wait);
}

void lightRec(uint16_t b,uint16_t c,uint16_t wait){
    
    strip.setPixelColor(b, c); // Set new pixel 'on'
    strip.show();              // Refresh LED states
    strip.setPixelColor(b, 0); // Erase pixel, but don't refresh!
    delay(wait);
    if(b < strip.numPixels()-1 ){
      lightRec(b+1,c,wait);
    }
    strip.setPixelColor(b, strip.Color(0,0,120)); // Set new pixel 'on'
    strip.show();              // Refresh LED states
    strip.setPixelColor(b, 0); // Erase pixel, but don't refresh!
    delay(wait);
    strip.show(); // Refresh to turn off last pixel
}




