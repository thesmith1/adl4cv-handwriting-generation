import {Component, OnInit} from '@angular/core';
import {ToBackendService} from './services/to-backend.service';
import {UtilsService} from './services/utils.service';
import {GlobalVars} from './global-vars';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'Title of the application';
  private _globalVars: GlobalVars;
  private _text: string;
  private _concurrencyCnt: number;
  public selectedStyle: number;
  public currentLineBase64 = '';
  constructor(private _backend: ToBackendService,
              public utils: UtilsService) {
    this._globalVars = new GlobalVars();
  }

  ngOnInit(): void {
    this._text = '';
    this._concurrencyCnt = 0;
    this.selectedStyle = 0;
  }

  public onChange(inputText: string) {
    this._concurrencyCnt = this._concurrencyCnt + 1;
    this.utils.wait(this._globalVars.keyboardTimeout).then(() => {
      this._concurrencyCnt = this._concurrencyCnt - 1;
      if (this._concurrencyCnt === 0) {
        const index = this._text.length;
        this._text = inputText;
        this._backend.generateImageFromString(this._text, this.selectedStyle, index).subscribe((ret) => {
          this.currentLineBase64 = ret['current_line'];
        });
      }
    });
  }

  public reset() {
    this._text = '';
    this.currentLineBase64 = '';
  }
}
